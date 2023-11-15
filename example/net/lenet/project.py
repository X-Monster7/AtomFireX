"""


================
@Author: zhicun Zeng / Alan
@Date: 2023/11/11 11:21
================
"""

from collections import defaultdict

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import Adam, SGD, RMSprop
from torch.utils import data

from tqdm import trange


class Util:
    _optimizer = {'SGD': SGD, 'Adam': Adam, 'RMSProp': RMSprop}
    OPTIMIZERS = defaultdict(lambda: SGD)
    OPTIMIZERS.update(_optimizer)
    _loss = {'CELoss': CrossEntropyLoss, 'MSELoss': MSELoss}
    LOSS = defaultdict(lambda: CrossEntropyLoss)
    LOSS.update(_loss)

    @staticmethod
    def load_dataset_minst(workers, _batch_size):
        trans = transforms.ToTensor()
        train_set = datasets.FashionMNIST(root = '..\..\..\data', train = True, transform = trans, download = False)
        test_set = datasets.FashionMNIST(root = '..\..\..\data', train = False, transform = trans, download = False)
        return data.DataLoader(dataset = train_set, batch_size = _batch_size, num_workers = workers, shuffle = True), \
            data.DataLoader(dataset = test_set, num_workers = workers, batch_size = _batch_size, shuffle = True)

    @staticmethod
    def init_weight(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    @staticmethod
    def try_gpu(model):
        if torch.cuda.device_count() > 1:
            print("Use", torch.cuda.device_count(), 'gpus')
            model = nn.DataParallel(model)
            model = model.cuda()
        return model


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 对于块状结构的叠加起来的神经网络，可以先用列表[]装着，
        # 然后nn.Seq加载它的地址*，即self.net = nn.Seq(*layers)
        self.net = nn.Sequential(
            # (28 - 5 + 2 * 2) / 1 + 1 = 28
            nn.Conv2d(1, 6, kernel_size = (5, 5), padding = 2),
            nn.Sigmoid(),
            # (28 - 2) / 2 + 1 = 14
            nn.AvgPool2d(kernel_size = (2, 2), stride = 2),
            # 14 - 5 + 1 = 10
            nn.Conv2d(6, 16, kernel_size = (5, 5)),
            nn.Sigmoid(),
            # (10 - 2) / 2 + 1 = 4
            nn.AvgPool2d(kernel_size = (2, 2), stride = 2),
            # 反正是全连接层，直接压平就好了
            nn.Flatten(),
            # 全连接层，前一层的全部参数都需要连接到每一个节点
            nn.Linear(16 * 5 * 5, 120),
            # 线性层之后也要激活函数呀！
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10),
        )

        def init_weight(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.xavier_uniform_(m.weight)

        self.net.apply(init_weight)

    def forward(self, x):
        x.view(-1, 1, 28, 28)
        return self.net.forward(x)


class TrainFrame:
    def __init__(self, net = LeNet, optim = 'Adam', lr = 0.1, loss = 'CELoss'):
        if not issubclass(net, nn.Module):
            raise Exception("非神经网络！无法训练！")
        self.net = net()
        self.optim = Util.OPTIMIZERS[optim](self.net.parameters(), lr = lr)
        self.loss = Util.LOSS[loss]()

    def train(self, epochs, batch_size = 128, workers = 4, device = None):
        train_iter, test_iter = Util.load_dataset_minst(workers, batch_size)
        # Util.try_gpu(self.net)
        # train_iter.__len__()是在分批的基础上的长度
        # 使用 trange 创建一个进度条，并指定总的 epoch 次数
        self.net.train()
        for epoch in trange(epochs, desc = 'Epoch', colour = 'blue'):
            for batch, (x, y) in enumerate(train_iter):
                # 在每个 epoch 内部，使用 trange 创建一个进度条，并指定迭代次数
                print(x.shape, y.shape)
                self.optim.zero_grad()
                y_pred = self.net(x)
                loss = self.loss(y_pred, y)
                loss.backward()
                self.optim.step()
                if batch % 5 == 0 or batch == len(train_iter) - 2:
                    print(f'Epoch: {epoch}, Batch: {batch}, Loss: {loss.item():.4f}')


my_train_frame = TrainFrame(net = LeNet, optim = 'Adam', lr = 0.01, loss = 'CELoss')
my_train_frame.train(epochs = 100, batch_size = 64, workers = 0)
