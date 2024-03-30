import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

dataset = datasets.ImageFolder(root="./dataset/raw-img")
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=1)


def calculate_mean_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in dataloader:
        # [B, C, W, H] => [B, C, W * H]
        data = data.view(data.size(0), data.size(1), -1)

        channels_sum += torch.mean(data, dim=[0, 2])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2) ** 0.5

    return mean, std


mean, std = calculate_mean_std(dataloader)
print(f"Mean: {mean}")
print(f"Std: {std}")
