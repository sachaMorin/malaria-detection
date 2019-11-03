"""Model and block definitions."""

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


class ConvBlock2(nn.Module):
    """Stack of 2 ConvBlocks.

    Kernel size can be set to 3 or 5.
    Max pooling only on last layer.
    """
    def __init__(self, channels_in, channels_out, kernel_size, dropout):
        super(ConvBlock2, self).__init__()
        self.conv1 = ConvBlock(channels_in, channels_out, kernel_size,
                               dropout, max_pool=False)
        self.conv2 = ConvBlock(channels_out, channels_out, kernel_size,
                               dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class ConvBlock3(nn.Module):
    """Stack of 3 ConvBlocks.

    Kernel size can be set to 3 or 5.
    Max pooling only on last layer.
    """
    def __init__(self, channels_in, channels_out, kernel_size, dropout):
        super(ConvBlock3, self).__init__()
        self.conv1 = ConvBlock(channels_in, channels_out, kernel_size,
                               dropout, max_pool=False)
        self.conv2 = ConvBlock(channels_out, channels_out, kernel_size,
                               dropout, max_pool=False)
        self.conv3 = ConvBlock(channels_out, channels_out, kernel_size,
                               dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

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

    def __init__(self, dp_conv, dp_fc):
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


class CustomVGG(nn.Module):
    """Loosely based on VGG."""

    def __init__(self, dp_conv, dp_fc):
        super(CustomVGG, self).__init__()
        self.conv1 = ConvBlock2(3, 64, 3, dp_conv)
        self.conv2 = ConvBlock3(64, 128, 3, dp_conv)
        self.conv3 = ConvBlock3(128, 256, 3, dp_conv)

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

        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x


class CustomInception(nn.Module):
    """Loosely based on Inception."""

    def __init__(self, dp_conv, dp_fc):
        super(CustomInception, self).__init__()
        self.conv1_1 = ConvBlock(3, 32, 3, dp_conv)
        self.conv1_2 = ConvBlock2(3, 32, 3, dp_conv)
        self.conv2_1 = ConvBlock(64, 64, 3, dp_conv)
        self.conv2_2 = ConvBlock2(64, 64, 3, dp_conv)
        self.conv3_1 = ConvBlock(128, 128, 3, dp_conv)
        self.conv3_2 = ConvBlock2(128, 128, 3, dp_conv)

        self.avg = nn.AvgPool2d(12)

        self.fc1 = nn.Linear(256, 256)
        self.dp_fc1 = nn.Dropout(dp_fc)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x_1 = self.conv1_1(x)
        x_2 = self.conv1_2(x)
        x = torch.cat((x_1, x_2), 1)
        x_1 = self.conv2_1(x)
        x_2 = self.conv2_2(x)
        x = torch.cat((x_1, x_2), 1)
        x_1 = self.conv3_1(x)
        x_2 = self.conv3_2(x)
        x = torch.cat((x_1, x_2), 1)

        x = self.avg(x)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x


class Tiny(nn.Module):
    """Tiny model to test pipeline on CPU on a poor laptop with no GPU."""

    def __init__(self, dp_conv, dp_fc):
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
