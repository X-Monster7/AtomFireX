# -*- coding: utf-8 -*-

"""
@author: Alan Tsang / Zeng Zhicun
@institution: CSU, China, changsha
@date: 2023/9/18 11:01
@version: 1.0.0
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process as Mp

import os

"""
standalone 是 torchrun 命令的一个选项，它指定了训练脚本是否应该在独立模式下运行。
如果您使用 --standalone 选项，则训练脚本将在独立模式下运行，这意味着它将在单独的进程中运行，而不是在 TorchElastic 群集中运行。
这个选项通常用于在单个节点上进行训练，而不是在多个节点上进行训练。
"""
"""
torchrun --standalone --nnodes= --nproc-per-node= YOUR_TRAINING_SCRIPT.py
"""


def mp_launch(function, world_size, args):
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(args.backend, init_method = args.dist_url)

    mp.spawn(function, args = (world_size,), nprocs = world_size, join = True)


def cmd_launch(function):
    os.system("launch %s" % function)
