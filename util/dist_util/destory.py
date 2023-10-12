# -*- coding: utf-8 -*-

"""
@author: Alan Tsang / Zeng Zhicun
@institution: CSU, China, changsha
@date: 2023/9/18 11:25
@version: 1.0.0
"""

from torch import distributed as dist


def clean_up():
    dist.destroy_process_group()
