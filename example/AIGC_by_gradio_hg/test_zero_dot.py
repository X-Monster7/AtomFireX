#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: Alan Tsang / Zeng Zhicun
@institution: CSU, China, changsha
@date: 2023/10/13 20:12
@version: 1.0.0
"""

import torch
import time

# x = torch.randn((1000, 100), dtype = torch.float32, device = 'cuda:0')
# y = torch.randn((1000, 100), dtype = torch.float32, device = 'cuda:0')
# start = time.time()
# for i in range(1000):
#     x * y
# end = time.time()
# print(end - start)

start = time.time()
x = torch.zeros((1000, 100), dtype = torch.float32, device = 'cuda:0')
y = torch.zeros((1000, 100), dtype = torch.float32, device = 'cuda:0')
start = time.time()
for i in range(1000):
    x * y
end = time.time()
print(end - start)
