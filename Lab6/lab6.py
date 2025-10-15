import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Subset,Dataset

import numpy as np
import random
import os
import cv2
import ray
from ray import tune
from ray.air import session
from skimage.util import random_noise
from sklearn.model_selection import train_test_split
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

class CustomImageDataset(Dataset):
    def __init__(self, image_paths,gauss_noise=False,gauss_blur=None,resize=128,p=0.5):
        self.p = p
        self.resize = resize
        self.gauss_noise = gauss_noise
        self.gauss_blur = gauss_blur
        # Remove self.center_crop since it's not defined or used
        # If you want to use center crop, you can define a transform like below:
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
        # Read image as RGB
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Resize and convert to tensor (ground truth)
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
            mean = random.uniform(-50, 50) / 255.0
            std = random.uniform(0.01, 0.1)
            img_np = noisy_image.permute(1,2,0).numpy()
            img_np = random_noise(img_np, mode='gaussian', mean=mean, var=std**2)
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
    
    # def save_model(self,name="model.pth"):
    #     torch.save(self.state_dict(), os.path.join(self.path, name))



data_dir = os.path.abspath("./img_align_celeba")  # Use absolute path

files = os.listdir(data_dir)
files = [os.path.join(data_dir, file) for file in files]


train_files, test_files = train_test_split(files[:100], test_size=0.2, random_state=42)


train_dataset = CustomImageDataset(train_files)
test_dataset = CustomImageDataset(test_files)

ray.shutdown()

def train_raytune(config):
    # Load dataset
    
    trainloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = Autoencoder(channels=config['channels']).to(device)
    loss_fn = nn.MSELoss()

    if config['optimizer'] == 'Adam':
        opt = optim.Adam(model.parameters(), lr=config['lr'])
    elif config['optimizer'] == 'SGD':
        opt = optim.SGD(model.parameters(), lr=config['lr'])
    
    for epoch in range(config['num_epochs']):
        model.train()
        avg_train_loss = 0.0
        avg_test_loss = 0.0
        for images, gt_images in trainloader:
            images, gt_images = images.to(device), gt_images.to(device)
            opt.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, gt_images)
            loss.backward()
            opt.step()
            avg_train_loss += loss.item() * images.size(0)
        avg_train_loss /= len(trainloader)

        total_pnsr = 0.0
        total_ssim = 0.0
        model.eval()
        for images, gt_images in testloader:
            images, gt_images = images.to(device), gt_images.to(device)
            with torch.no_grad():
                outputs = model(images)
                loss = loss_fn(outputs, gt_images)
                avg_test_loss += loss.item() * images.size(0)
                # Calculate PSNR and SSIM
                for i in range(images.size(0)):
                    output_img = outputs[i].cpu().numpy().transpose(1, 2, 0)
                    gt_img = gt_images[i].cpu().numpy().transpose(1, 2, 0)
                    total_pnsr += psnr(gt_img, output_img, data_range=1.0)
                    total_ssim += ssim(gt_img, output_img, data_range=1.0, channel_axis=2)

        avg_test_loss /= len(testloader)
        total_pnsr /= len(testloader)
        total_ssim /= len(testloader)

        session.report({
            "train_loss": avg_train_loss,
            "val_loss": avg_test_loss,
            "val_psnr": total_pnsr,
            "val_ssim": total_ssim
        })

ray.init(num_gpus=1)
config = {
    'channels': tune.grid_search([[32, 64, 128], [64,128,256], [64, 128, 256, 512]]),
    'batch_size': tune.grid_search([16, 32]),
    'optimizer': tune.grid_search(['Adam', 'SGD']),
    'lr': tune.grid_search([1e-3, 8e-4, 1e-4, 1e-2]),
    'num_epochs': tune.grid_search([10, 50, 100])
}

from ray.tune import RunConfig

def short_trial_name(trial):
    return f"trial_{trial.trial_id}"

from ray.tune import TuneConfig

tuner = tune.Tuner(
    tune.with_resources(
        train_raytune,
        resources={ "gpu": 0.33}
    ),
    param_space=config,
    run_config=RunConfig(
        storage_path=os.path.abspath("./ray_results")
    ),
    tune_config=TuneConfig(
        trial_dirname_creator=short_trial_name,
        metric="val_psnr",
        mode='max',
    )
)

result = tuner.fit()

# save best config in to text file
path = os.path.abspath('.')
best_result = result.get_best_result(metric="val_psnr", mode='max')
with open(os.path.join(path, "grid_best_config.txt"), "w") as f:
    f.write(str(best_result.config))
    f.close()
    
ray.shutdown()

ray.init(num_gpus=1)
config = {
    "path": "d:/Image_Processing/Lab6/models/random",  # Use absolute path
    "channels": tune.choice([[32, 64, 128], [64, 128, 256], [64, 128, 256, 512]]),
    "lr": tune.uniform(1e-4, 1e-2),
    "batch_size": tune.randint(16, 33),  # randint is inclusive of low, exclusive of high
    "num_epochs": tune.randint(10, 101),
    "optimizer": tune.choice(["Adam", "SGD"])
}

tune_config = tune.TuneConfig(
    num_samples=4,  # Random search with 80 samples
    trial_dirname_creator=short_trial_name,
    metric="val_psnr",
    mode="max"
)

tuner = tune.Tuner(
    tune.with_resources(
        train_raytune,
        resources={ "gpu": 0.33}
    ),
    param_space=config,
    run_config=RunConfig(
        storage_path=os.path.abspath("./ray_results")
    ),
    tune_config=tune_config
)

result = tuner.fit()

path = os.path.abspath('.')
best_result = result.get_best_result(metric="val_psnr", mode='max')
with open(os.path.join(path, "random_best_config.txt"), "w") as f:
    f.write(str(best_result.config))
    f.close()
