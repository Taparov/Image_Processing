import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Subset,Dataset

import matplotlib.pyplot as plt
import numpy as np
import random
import os
import cv2

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
    
model = Autoencoder([32, 64, 128])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = Autoencoder([64, 128, 256, 512])
model.load_state_dict(torch.load("./models/grid/model_s53432_00015_f1.pt"))
model.to(device)
model.eval()

# load images from ./test and save the output with same file name to ./test_results
input_folder = "./test_real"
output_folder = "./test"

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = model(input_tensor)

        output_image = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output_image = np.clip(output_image, 0, 1)  # Ensure values are in [0, 1] for matplotlib
        plt.imsave(os.path.join(output_folder, filename), output_image)
