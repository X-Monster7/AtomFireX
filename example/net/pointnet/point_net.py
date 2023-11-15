"""pointnet Implement By Pytorch

Author: Alan Tsang / Zeng Zhicun
Institution: CSU, China, changsha
Date: 2023年10月25日

"""

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

from pprint import pprint

torch.manual_seed(1)


class STN3d(nn.Module):
    """
    STN: spatial transform network
    3d: STN3d output (3, 3) matrix

    Supply(C-Sec 5.1): It’s composed of a shared MLP (64, 128, 1024) network ,
    a max pooling across points and two fully connected layers with output sizes 512, 256.
    The output matrix is initialized as an identity matrix.
    All layers, except the last one, include ReLU and batch normalization.

    MLP can not only be implemented by Conv, but also Linear or FC
    """

    def __init__(self):
        super(STN3d, self).__init__()
        self.conv64 = nn.Conv1d(3, 64, 1)
        self.conv128 = nn.Conv1d(64, 128, 1)
        self.conv1024 = nn.Conv1d(128, 1024, 1)
        # fc: full connect全连接层
        self.fc512 = nn.Linear(1024, 512)
        self.fc256 = nn.Linear(512, 256)
        self.fc9 = nn.Linear(256, 9)

        self.relu = nn.ReLU()

        self.bn64 = nn.BatchNorm1d(64)
        self.bn128 = nn.BatchNorm1d(128)
        self.bn1024 = nn.BatchNorm1d(1024)
        self.bn512 = nn.BatchNorm1d(512)
        self.bn256 = nn.BatchNorm1d(256)

    def forward(self, x):
        """
        点云转为变换矩阵。
        Args:
            x: (batch_size,  3, num_points)

        Returns: (3, 3)的identity matrix

        """
        batch_size = x.shape[0]
        x = F.relu(self.bn64(self.conv64(x)))
        x = F.relu(self.bn128(self.conv128(x)))
        x = F.relu(self.bn1024(self.conv1024(x)))
        # (batch, 1024, num_point)
        # max return (value, slice)
        x = torch.max(x, 2, keepdim = False)[0]
        # view can squeeze tensor like torch.squeeze
        # x = x.view(-1, 1024)
        x = self.bn512(self.bn512(self.fc512(x)))
        x = self.bn256(self.bn256(self.fc256(x)))
        x = self.fc9(x)

        identity = (torch.from_numpy(
            np.eye(3, dtype = np.float32)
            .flatten()
        ).repeat(batch_size, 1))
        if x.is_cuda:
            identity = identity.cuda()
        x = (x + identity).reshape(-1, 3, 3)
        return x


class STNkd(nn.Module):
    """
    STN: spatial transform network in Paper Supply(C-Sec 5.1)

    The composition of the STNkd is similar to that of the STN3d,
    the only difference being the shape of the inputs and outputs.

    MLP can not only be implemented by Conv, but also Linear or FC
    """

    def __init__(self, k = 64):
        """

        Args:
            k ():
        """
        super(STNkd, self).__init__()
        self.k = k
        self.conv64 = nn.Conv1d(k, 64, 1)
        self.conv128 = nn.Conv1d(64, 128, 1)
        self.conv1024 = nn.Conv1d(128, 1024, 1)
        # fc: full connect全连接层
        self.fc512 = nn.Linear(1024, 512)
        self.fc256 = nn.Linear(512, 256)
        self.fc9 = nn.Linear(256, k * k)

        self.bn64 = nn.BatchNorm1d(64)
        self.bn128 = nn.BatchNorm1d(128)
        self.bn1024 = nn.BatchNorm1d(1024)
        self.bn512 = nn.BatchNorm1d(512)
        self.bn256 = nn.BatchNorm1d(256)

    def forward(self, x):
        """
        提取点云数据的特征
        Args:
            x: (batch_size,  3, 64)

        Returns: (64, 64)的identity matrix

        """
        batch_size = x.shape[0]
        x = F.relu(self.bn64(self.conv64(x)))
        x = F.relu(self.bn128(self.conv128(x)))
        x = F.relu(self.bn1024(self.conv1024(x)))
        # (batch, 1024, num_point)
        # max return (value, slice)
        x = torch.max(x, 2, keepdim = False)[0]
        # view can squeeze tensor like torch.squeeze
        # x = x.view(-1, 1024)
        x = self.bn512(self.bn512(self.fc512(x)))
        x = self.bn256(self.bn256(self.fc256(x)))
        x = self.fc9(x)

        identity = torch.from_numpy(
            np.eye(
                self.k,
                dtype = np.float32
            ).flatten()
        ).repeat(
            batch_size,
            1
        )
        if x.is_cuda:
            identity = identity.cuda()
        x = (x + identity).reshape(-1, self.k, self.k)
        return x


class PointEncoder(nn.Module):
    """
    return transformed feature.
    """

    def __init__(self):
        super(PointEncoder, self).__init__()
        self.stn = STN3d()
        self.stn_f = STNkd(k = 64)

        self.conv64 = torch.nn.Conv1d(3, 64, 1)
        self.cov64_ = torch.nn.Conv1d(64, 64, 1)
        self.conv128 = torch.nn.Conv1d(64, 128, 1)
        self.conv1024 = torch.nn.Conv1d(128, 1024, 1)

        self.bn64 = nn.BatchNorm1d(64)
        self.bn64_ = nn.BatchNorm1d(64)
        self.bn128 = nn.BatchNorm1d(128)
        self.bn1024 = nn.BatchNorm1d(1024)

    def forward(self, x):
        n_pts = x.size()[2]
        # T-NET 1
        input_trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, input_trans)
        x = x.transpose(2, 1)

        # MLP 1
        x = F.relu(self.bn64(self.conv64(x)))
        x = F.relu(self.bn64_(self.cov64_(x)))

        # T-NET 2
        feat_trans = self.stn_f(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, feat_trans)
        x = x.transpose(2, 1)

        # MLP 2
        x = F.relu(self.bn128(self.conv128(x)))
        x = self.bn1024(self.conv1024(x))

        # MAX POOL
        x = torch.max(x, 2, keepdim = True)[0]
        x = x.view(-1, 1024)
        return x


# class PointNetCls(nn.Module):
#     def __init__(self, k = 2):
#         super(PointNetCls, self).__init__()
#         self.feat = PointEncoder()
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, k)
#         self.dropout = nn.Dropout(p = 0.3)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.feat(x)
#         encode_feature = x
#         x = F.relu(self.bn1(self.fc1(x)))
#         x = F.relu(self.bn2(self.dropout(self.fc2(x))))
#         x = self.fc3(x)
#         return F.log_softmax(x, dim = 1), encode_feature


if __name__ == '__main__':
    stn = PointEncoder()
    x = torch.randn((16, 3, 1010))
    pprint(stn(x).shape)
