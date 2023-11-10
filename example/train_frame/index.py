import torch.optim
import yaml

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from example.train_frame.trainer import Trainer
from example.train_frame.hook import Saver, Writer
from example.net.lenet.index import LeNet

train_set = FashionMNIST(
    root = r'../../data',
    train = True,
    transform = transforms.ToTensor(),
    download = True
)
# test_set = FashionMNIST(root = '', train = False, transform = trans, download = True)
with open('./config/config.yml') as _:
    config = yaml.safe_load(_)
dataloader = DataLoader(dataset = train_set, batch_size = config['train']['batch_size'])
net = LeNet()
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr = config['train']['lr'])

trainer = Trainer(
    data_loader = dataloader,
    model = net,
    loss = loss,
    optimizer = optimizer,
    config = config
)

hooks = [Saver(save_per_epoch = 5), Writer(save_per_epoch = 5)]
trainer.register_hooks(hooks)
trainer.fit()
