import yaml
import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST

# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上两级目录的绝对路径
two_levels_up = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, two_levels_up)

from example.net.lenet.index import LeNet
from util.dist_util import init, function, launch, destory
from hook import Saver, Writer
from trainer import Trainer

# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 切换当前工作目录到项目目录
os.chdir(current_dir)
train_set = FashionMNIST(
    root = r'../../data',
    train = True,
    transform = transforms.ToTensor(),
    download = True
)
# test_set = FashionMNIST(root = '', train = False, transform = trans, download = True)
with open('config/config.yml') as _:
    config = yaml.safe_load(_)

# setup = {'backend': 'gloo'}
# TODO: 实现设置环境的传参，例如backend和url等
args = init.setup_environment()

dataloader = function.DDPDataLoader(dataset = train_set, batch_size = config['train']['batch_size'])
net = function.DDPNet(LeNet().cuda(), args)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr = config['train']['lr'])

trainer = Trainer(
    data_loader = dataloader,
    model = net,
    loss = loss,
    optimizer = optimizer,
    config = config
)

# 这一部分可以作为插件，使用配置文件开启关闭，并配置路径等参数，注意设置默认值。
hooks = [Saver(save_per_epoch = 5), Writer(save_per_epoch = 2)]
# =============================================================
trainer.register_hooks(hooks)
trainer.fit()
destory.clean_up()
