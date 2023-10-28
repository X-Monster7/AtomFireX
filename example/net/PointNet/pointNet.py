#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

@description:

==========================================
@author: Alan Tsang / Zeng Zhicun
@institution: CSU, China, changsha
@date: 2023/10/23 8:10
@version: 1.0.0
"""

from util.logger_util.index import Log
import torch
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, Linear, BatchNorm2d, Dropout, ReLU, Softmax, Sequential


class PointNet(nn.Module):
    def __init__(self, num_point, ):
        super(PointNet, self).__init__()
        self.num_point = num_point
        # ============== T-Net 1
        self.input_transform = Sequential(
            Conv2d(1, 64, (1, 3)),
            # 参数是特征维度
            BatchNorm2d(64),
            ReLU(),
            Conv2d(64, 128, (1, 1)),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(128, 1024, (1, 1)),
            BatchNorm2d(1024),
            ReLU(),
            MaxPool2d((num_point, 1))
        )
        self.input_fc = Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 9)
        )
        # Initialize weights and bias as specified
        self.input_fc[-1].weight.data = torch.zeros(256, 9)
        self.input_fc[-1].bias.data = torch.eye(3).view(-1)
        # ==== end === T-net 1 =================

        self.mlp_1 = Sequential(
            Conv2d(1, 64, (1, 3)),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(64, 64, (1, 1)),
            BatchNorm2d(64),
            ReLU(),
        )
        self.feature_transform = Sequential(
            Conv2d(64, 64, (1, 1)),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(64, 128, (1, 1)),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(128, 1024, (1, 1)),
            BatchNorm2d(1024),
            ReLU(),
            MaxPool2d((num_point, 1))
        )
        self.feature_fc = Sequential(
            Linear(1024, 512),
            ReLU(),
            Linear(512, 256),
            ReLU(),
            Linear(256, 64 * 64)
        )
        self.mlp_2 = Sequential(
            Conv2d(64, 64, (1, 1)),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(64, 128, (1, 1)),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(128, 1024, (1, 1)),
            BatchNorm2d(1024),
            ReLU(),
        )

    def forward(self, inputs):
        batch_size = inputs.shape[0]

        t_net = self.input_transform(inputs)
        t_net = torch.squeeze(t_net)
        t_net = self.input_fc(t_net)
        t_net = torch.reshape(t_net, [batch_size, 3, 3])

        x = torch.reshape(inputs, shape = (batch_size, 1024, 3))
        # [batch_size, 1024, 3]
        x = torch.matmul(x, t_net)
        x = torch.unsqueeze(x, dim = 1)
        x = self.mlp_1(x)

        t_net = self.feature_transform(x)
        t_net = torch.squeeze(t_net)
        t_net = self.feature_fc(t_net)
        t_net = torch.reshape(t_net, [batch_size, 64, 64])

        x = torch.reshape(x, shape = (batch_size, 64, 1024))
        x = torch.transpose(x, 1, 2)
        x = torch.matmul(x, t_net)
        x = torch.transpose(x, 1, 2)
        x = torch.unsqueeze(x, dim = -1)
        point_feat = x
        x = self.mlp_2(x)
        x = torch.max(x, dim = 2)

        global_feat_expand = torch.tile(torch.unsqueeze(x, dim = 1), [1, self.num_point, 1, 1])
        x = torch.concat([point_feat, global_feat_expand], dim = 1)
        x = self.seg_net(x)
        x = torch.squeeze(x, dim = -1)
        x = torch.transpose(x, 1, 2)

        return x
