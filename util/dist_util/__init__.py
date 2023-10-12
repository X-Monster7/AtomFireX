# -*- coding: utf-8 -*-

"""
@author: Alan Tsang / Zeng Zhicun
@institution: CSU, China, changsha
@date: 2023/9/18 8:48
@version: 1.0.0
"""

import os
import sys
import torch.cuda
import torch.distributed as dist
import argparse

__version__ = '1.0.0'
DistTool = sys.modules[__name__]
