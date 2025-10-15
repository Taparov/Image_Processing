import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Subset,Dataset

import numpy as np
import random
import os
import cv2
from skimage.util import random_noise
from sklearn.model_selection import train_test_split
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, gauss_noise=False, gauss_blur=None, resize=128, p=0.5, noise_factor=0.0, noise_std=0.0):
        self.p = p
        self.resize = resize
        self.gauss_noise = gauss_noise
        self.gauss_blur = gauss_blur
        self.noise_factor = float(noise_factor)
        self.noise_std = float(noise_std)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor()
        ])
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt_image = self.transform(image)
        noisy_image = gt_image.clone()
        # Apply Gaussian blur with probability p
        if self.gauss_blur and random.random() < self.p:
            ksize = random.choice([k for k in range(3, 12, 2)])
            img_np = gt_image.permute(1,2,0).numpy() * 255
            img_np = img_np.astype(np.uint8)
            img_np = cv2.GaussianBlur(img_np, (ksize, ksize), 0)
            noisy_image = torch.from_numpy(img_np).permute(2,0,1).float() / 255.0
        # Apply Gaussian noise with probability p
        if self.gauss_noise and random.random() < self.p:
            mean = self.noise_factor
            std = self.noise_std
            img_np = noisy_image.permute(1,2,0).cpu().numpy()
            # noise = np.random.normal(loc=mean, scale=std, size=img_np.shape)
            noise = self.noise_factor * np.random.normal(loc=0.0, scale=std, size=img_np.shape)
            img_np = img_np + noise
            img_np = np.clip(img_np, 0, 1)
            noisy_image = torch.from_numpy(img_np).permute(2,0,1).float()
        return noisy_image, gt_image
    

class DownSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DownSamplingBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class UpSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(UpSamplingBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.upsample(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, channels=[64, 128, 256], input_channels=3, output_channels=3):
        super().__init__()

        self.channels = channels

        # Input Layer
        self.input_conv = nn.Conv2d(input_channels, channels[0], kernel_size=3, stride=1, padding=1)

        for i in range(len(channels)-1):
            setattr(self, f'down{i+1}', DownSamplingBlock(channels[i], channels[i+1]))
        for i in range(len(channels)-1, 0, -1):
            setattr(self, f'up{i}', UpSamplingBlock(channels[i], channels[i-1]))


        # Output Layer
        self.output_conv = nn.Conv2d(channels[0], output_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        # skip_connections = []
        x = self.input_conv(x)
        for i in range(len(self.channels)-1):
            x = getattr(self, f'down{i+1}')(x)
            # skip_connections.append(x)
        for i in range(len(self.channels)-1, 0, -1):
            x = getattr(self, f'up{i}')(x)
            # x += skip_connections[i-1]
        x = self.output_conv(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = Autoencoder([64, 128, 256, 512])
model.load_state_dict(torch.load("./models/grid/model_s53432_00015_f1.pt"))
model.to(device)
model.eval()

test_data_dir = "./img_align_celeba/"
files = os.listdir(test_data_dir)
files = [os.path.join(test_data_dir, file) for file in files]

test_dataset = CustomImageDataset(files[10000:11000], gauss_noise=True, gauss_blur=True, resize=128, p=0.5, noise_factor=0.2, noise_std=0.1)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def evaluate_model(model, dataloader):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    with torch.no_grad():
        for noisy_imgs, gt_imgs in dataloader:
            noisy_imgs = noisy_imgs.to(device)
            gt_imgs = gt_imgs.to(device)
            outputs = model(noisy_imgs)
            outputs = outputs.clamp(0, 1)  # Ensure outputs are in valid range

            for i in range(outputs.size(0)):
                output_img = outputs[i].permute(1, 2, 0).cpu().numpy()
                gt_img = gt_imgs[i].permute(1, 2, 0).cpu().numpy()

                total_psnr += psnr(gt_img, output_img, data_range=1.0)
                total_ssim += ssim(gt_img, output_img, data_range=1.0, channel_axis=2)
            count += outputs.size(0)

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    return avg_psnr, avg_ssim

print("Evaluating model...")
avg_psnr, avg_ssim = evaluate_model(model, test_loader)
print(f"Average PSNR: {avg_psnr:.2f}")
print(f"Average SSIM: {avg_ssim:.4f}")