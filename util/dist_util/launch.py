# -*- coding: utf-8 -*-

"""
@author: Alan Tsang / Zeng Zhicun
@institution: CSU, China, changsha
@date: 2023/9/18 11:01
@version: 1.0.0
"""

import torch.multiprocessing as mp
import os


def mp_launch(function, world_size):
    mp.spawn(function, args=(world_size,), nprocs=world_size, join=True)


def cmd_launch(function):
    os.system("launch %s" % function)
