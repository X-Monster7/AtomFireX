""" The Implement of resnet.

Author: Alan
Date: 2023年10月26日
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels = in_channel, out_channels = out_channel,
            kernel_size = (3, 3), stride = 2, padding = 1
        )
        self.conv2 = nn.Conv2d(
            in_channels = out_channel, out_channels = out_channel,
            kernel_size = (3, 3), stride = 2, padding = 1
        )
        if in_channel != out_channel:
            self.conv3 = nn.Conv2d(in_channel, out_channel, (1, 1), stride = 2)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        origin = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.conv3:
            return F.relu(self.conv3(origin) + x)

        return F.relu(x + origin)
