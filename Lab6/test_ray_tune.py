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
import glob
import ray
from ray import tune, air
from ray.air import session
from skimage.util import random_noise
from sklearn.model_selection import train_test_split, KFold
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

seed = 4912
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

ray.shutdown()

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
    def __init__(self, path, channels=[64, 128, 256], input_channels=3, output_channels=3):
        super().__init__()
        # Input Layer
        self.channels = channels
        self.path = path
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
    
    def save(self,name):
        path = os.path.join(self.path, name)
        torch.save(self.state_dict(), path)

data_dir = os.path.abspath("./img_align_celeba")

files = os.listdir(data_dir)
files = [os.path.join(data_dir, file) for file in files]


train_files, test_files = train_test_split(files[:1000], test_size=0.2, random_state=42)


train_dataset = CustomImageDataset(train_files, gauss_noise=True, gauss_blur=True, resize=128, p=0.5, noise_factor=0.3, noise_std=0.5)
test_dataset = CustomImageDataset(test_files, gauss_noise=True, gauss_blur=True, resize=128, p=0.5, noise_factor=0.3, noise_std=0.5)

print("cuda" if torch.cuda.is_available() else "cpu")

kf = KFold(n_splits=2, shuffle=True, random_state=42)
folds = {}
for fold, (train_idx, valid_idx) in enumerate(kf.split(np.arange(len(train_dataset)))):
    folds[fold] = {
        "train_idx": train_idx,
        "valid_idx": valid_idx
    }

def train_raytune(config):


    
    train_subset = Subset(train_dataset, folds[config["fold"]]["train_idx"])
    valid_subset = Subset(train_dataset, folds[config["fold"]]["valid_idx"])

    trainloader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True)
    validloader = DataLoader(valid_subset, batch_size=config['batch_size'], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(path=config['path'], channels=config['channels']).to(device)
    loss_fn = nn.MSELoss()

    if config['optimizer'] == 'Adam':
        opt = optim.Adam(model.parameters(), lr=config['lr'])
    elif config['optimizer'] == 'RMSProp':
        opt = optim.RMSprop(model.parameters(), lr=config['lr'], momentum=0.9)

    for epoch in range(config['num_epochs']):
        model.train()
        avg_train_loss = 0.0
        loss_total = 0
        for images, gt_images in trainloader:
            images, gt_images = images.to(device), gt_images.to(device)
            opt.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, gt_images)
            loss.backward()
            opt.step()
            avg_train_loss += loss.item() * images.size(0)
            loss_total += images.size(0)
        avg_train_loss /= loss_total

        total_pnsr = 0.0
        total_ssim = 0.0
        avg_valid_loss = 0.0
        valid_total = 0
        model.eval()
        for images, gt_images in validloader:
            images, gt_images = images.to(device), gt_images.to(device)
            with torch.no_grad():
                outputs = model(images)
                loss = loss_fn(outputs, gt_images)
                avg_valid_loss += loss.item() * images.size(0)
                # Calculate PSNR and SSIM
                for i in range(images.size(0)):
                    output_img = outputs[i].cpu().numpy().transpose(1, 2, 0)
                    gt_img = gt_images[i].cpu().numpy().transpose(1, 2, 0)
                    total_pnsr += psnr(gt_img, output_img, data_range=1.0)
                    total_ssim += ssim(gt_img, output_img, data_range=1.0, channel_axis=2)
            valid_total += images.size(0)

        avg_valid_loss /= valid_total
        avg_pnsr = total_pnsr / valid_total
        avg_ssim = total_ssim / valid_total



        session.report({
            "train_loss": avg_train_loss,
            "val_loss": avg_valid_loss,
            "val_psnr": avg_pnsr,
            "val_ssim": avg_ssim,
        })
    model.save(f"model_s{session.get_trial_id()}_f{fold}.pt")

ray.init(num_gpus=1)
config = {
    'path': os.path.abspath("./models/grid/"),
    'fold': tune.grid_search([0, 1]),  # Assuming 2 folds for K-Fold Cross-Validation
    'channels': tune.grid_search([[64, 128, 256, 512]]),
    'batch_size': tune.grid_search([2, 16]),
    'optimizer': tune.grid_search(['RMSProp']),
    'lr': tune.grid_search([0.01, 0.0001]),
    'num_epochs': tune.grid_search([100, 300])
}
from ray.tune import RunConfig
from ray.tune.progress_reporter import CLIReporter

def short_trial_name(trial):
    return f"trial_{trial.trial_id}"

from ray.tune import TuneConfig

reporter = CLIReporter()
reporter._report_interval = 120  # Set report interval to 120 seconds (2 minutes)

tuner = tune.Tuner(
    tune.with_resources(
        train_raytune,
        resources={ "cpu": 1, "gpu": 0.5 }
    ),
    param_space=config,
    run_config=RunConfig(
        storage_path=os.path.abspath("./ray_results"),
        progress_reporter=reporter
    ),
    tune_config=TuneConfig(
        trial_dirname_creator=short_trial_name,
        metric="val_psnr",
        mode='max',
    )
)

result = tuner.fit()

print("🎉[INFO] Training is done!")
print("Best config is:", result.get_best_result().config)
print("Best result is:", result.get_best_result())
df = result.get_dataframe()
df.to_csv('./ray_results.csv', index=False)