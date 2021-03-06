"""Model and block definitions."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Building blocks
class ConvBlock(nn.Module):
    """Standard convolution 'block'.

    Convolution block with batchnorm, dropout and optional max pooling.
    Kernel size can be set to 3 or 5.
    Same padding will be applied for stable input dimensions.
    """
    def __init__(self, channels_in, channels_out, kernel_size, dropout,
                 max_pool=True):
        super(ConvBlock, self).__init__()
        self.max_pool = max_pool

        # Same padding
        if kernel_size is 3:
            padding = 1
        elif kernel_size is 5:
            padding = 2
        else:
            raise Exception('Kernel_size should be 3 or 5.')

        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size,
                              padding=padding)
        self.bn = nn.BatchNorm2d(channels_out)
        self.dp = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        if self.max_pool:
            x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.bn(x)
        x = self.dp(x)

        return x
# Models
class PaperCNN(nn.Module):
    """Custom CNN based on paper.

    Model based on the 'Performance evaluation of deep neural ensembles toward
    malaria parasite detection in thin-blood smear images' paper
    from Sivaramakrishnan Rajaraman​, Stefan Jaeger and Sameer K. Antani.

    In short, 3 convolutionnal blocks followed by a GAP layer and 1
    fully-connected hidden layer.
    """

    def __init__(self, dp_conv=0, dp_fc=0):
        super(PaperCNN, self).__init__()
        self.conv1 = ConvBlock(3, 64, 5, dp_conv)
        self.conv2 = ConvBlock(64, 128, 5, dp_conv)
        self.conv3 = ConvBlock(128, 256, 5, dp_conv)

        self.avg = nn.AvgPool2d(12)

        self.fc1 = nn.Linear(256, 256)
        self.dp_fc1 = nn.Dropout(dp_fc)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.avg(x)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dp_fc1(x)

        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x

class Tiny(nn.Module):
    """Tiny model to test pipeline on CPU on a poor laptop with no GPU."""

    def __init__(self, dp_conv=0, dp_fc=0):
        super(Tiny, self).__init__()
        self.conv1 = ConvBlock(3, 2, 3, dp_conv)
        self.avg = nn.AvgPool2d(48)
        self.fc1 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.avg(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = torch.sigmoid(x)

        return x
