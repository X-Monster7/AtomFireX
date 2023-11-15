# -*- coding: utf-8 -*-

"""
@author: Alan Tsang / Zeng Zhicun
@institution: CSU, China, changsha
@date: 2023/9/18 18:23
@version: 1.0.0
"""
from typing import Callable

import torch
from torch import distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as DSP
from torch.nn.parallel import DistributedDataParallel as DDP


def DDPNet(net, args):
    return DDP(net, device_ids = [args.local_rank], output_device = args.local_rank)


def DDPDataLoader(dataset, batch_size: int, shuffle: bool = True):
    sampler = DSP(dataset, shuffle = shuffle)
    return DataLoader(dataset, batch_size, sampler = sampler)


def reduce_value(value, average = True):
    """
    该函数的作用是手动聚合并平均rank中的梯度数据。
    Args:
        value ():
        average ():

    Returns:

    """
    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= dist.get_world_size()

        return value


get_rank = dist.get_rank
get_world_size = dist.get_world_size
is_main_process: Callable[[], bool] = lambda: get_rank() == 0
