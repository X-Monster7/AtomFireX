# -*- coding: utf-8 -*-

"""
@author: Alan Tsang / Zeng Zhicun
@institution: CSU, China, changsha
@date: 2023/9/18 9:05
@version: 1.0.0

@date: 2023/11/15 16:06
@version: 1.1.0
@des: 增加torchrun，弄清楚了冗余的参数。
"""

import os
from argparse import Namespace

import torch.cuda
import torch.distributed as dist
import argparse


def _init_distributed_args(parameter: dict):
    backend = parameter.get('backend', 'nccl')
    dist_url = parameter.get('dist_url', 'env://')

    parser = argparse.ArgumentParser()
    # 是否启用SyncBatchNorm
    # parser.add_argument('--syncBN', type = bool, default = False)

    # parser.add_argument('--device', default = 'cuda', help = 'device id (i.e. 0 or 0,1 or cpu)')

    parser.add_argument('--backend', default = backend, help = 'nccl、gloo')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    # parser.add_argument(
    #     '--world-size', default = world_size, type = int,
    #     help = 'number of all distributed processes of all machine'
    # )
    parser.add_argument('--dist-url', default = dist_url, help = 'url used to set up distributed training')
    # 如果使用的是torchrun，那么master_addr和master_port都不用进行设置了
    # parser.add_argument('--master_addr', default = '127.0.0.1', help = 'master address for distributed training')
    # parser.add_argument('--master_port', default = '23471', help = 'master port for distributed training')

    # 如果使用的是torchrun，那么local_rank通过环境变量进行设置
    # parser.add_argument('--local_rank', type = int)
    args = parser.parse_args()
    args.distributed = True
    args.local_rank = int(os.environ["LOCAL_RANK"])

    return args


def setup_environment(parameter = None) -> Namespace:
    """
    通过os.environ环境变量设置分布式训练的基本参数
    Args:
        parameter: {’backend': 'nccl', 'master_addr': '127.0.0.1', 'master_port': '29500'}等

    Returns:
        None

    """
    if parameter is None:
        parameter = {}
    if not torch.cuda.is_available():
        raise EnvironmentError("not find GPU device for training.")

    args = _init_distributed_args(parameter)

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(args.backend, init_method = args.dist_url)

    return args
