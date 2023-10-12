# -*- coding: utf-8 -*-

"""
@author: Alan Tsang / Zeng Zhicun
@institution: CSU, China, changsha
@date: 2023/9/18 9:05
@version: 1.0.0
"""

import os

import torch.cuda
import torch.distributed as dist
import argparse


def init_distributed_mode_(args):
    torch.cuda.set_device(args.gpu)

    dist.init_process_group('nccl', init_method=args.dist_url)


def init_distributed_args_(world_size: int):
    parser = argparse.ArgumentParser()
    # 是否启用SyncBatchNorm
    parser.add_argument('--syncBN', type=bool, default=False)

    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world-size', default=world_size, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    return parser.parse_args()


def setup_environment(world_size: int , *args, **kwargs) -> None:
    """
    通过os.environ环境变量设置分布式训练的基本参数
    Args:
        world_size ():
        *args (): distribution training environment arguments,like rank、
        **kwargs ():

    Returns:
        None

    """
    if not torch.cuda.is_available():
        raise EnvironmentError("not find GPU device for training.")

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # SLURM：数据集群作业管理方案
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        raise Exception('Cant find necessary arguments before init distributions training!')
        args.distributed = False
        return

    args.distributed = True

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '23471'

    args = init_distributed_args_(world_size)
    init_distributed_mode_(args)
    print('distributed training, init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)

    dist.barrier()
    return args
