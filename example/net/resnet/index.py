""" The Implement of resnet.

Author: Alan
Date: 2023年10月26日
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_channel, out_channel, is_use_1plus1_conv):
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


# class ResNet18(nn.Module):
#     def __init__(self, input_channel, num_hidden, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.block1 = nn.Sequential(
#             nn.Conv2d(kernel_size = 7, stride = 2, padding = 3, in_channels = input_channel, out_channels = 64),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
#         )
#         self.block2 = Block(64, num_hidden)
#         self.block3 =
#         nn.AdaptiveAvgPool2d((1, 1)),
#         nn.Flatten(), nn.Linear(512, 10)

if __name__ == "__main__":
    blk = Block(3, 3)
    X = torch.rand(4, 3, 6, 6)
    Y = blk(X)
    print(Y.shape)
