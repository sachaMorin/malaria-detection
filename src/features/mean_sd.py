"""Find by channel mean and standard deviation on train set."""
import os

import torch
from torchvision import transforms, datasets
from src.data.CustomDataset import CustomDataset

TRAIN_PATH = os.path.join(
    os.path.dirname(__file__),
    '../../data/processed/train'
)

# Prep loader
data_transform = transforms.Compose(
    [transforms.Resize(size=(96, 96)), transforms.ToTensor()])
trainset = CustomDataset(root=TRAIN_PATH, transform=data_transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                           shuffle=False, num_workers=4)

# Variables to accumulate results
total = torch.tensor([0.0, 0.0, 0.0])
distance_squared = torch.tensor([0.0, 0.0, 0.0])

# Mean
for data in train_loader:
    inputs, _ = data
    for channel in range(3):
        total[channel] += torch.sum(inputs[:, channel]).item()

mean = total / (96 * 96 * len(trainset))
print('Mean:')
print(mean)


# Standard deviation
for data in train_loader:
    inputs, _ = data
    for channel in range(3):
        distance_squared[channel] += torch.sum(
            (inputs[:, channel] - mean[channel].item()) ** 2).item()

sd = (distance_squared / (96 * 96 * len(trainset))) ** 0.5
print('Sd:')
print(sd)
