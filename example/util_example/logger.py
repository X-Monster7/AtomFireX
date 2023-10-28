#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: Alan Tsang / Zeng Zhicun
@institution: CSU, China, changsha
@date: 2023/10/18 22:29
@version: 1.0.0
"""
import os
import sys

# 这行代码使得在命令行的无论哪一个目录中，都可以运行该文件
sys.path.append(f"E:\deeplearning\AtomFire")

from util.logger_util.index import Log

# 获取当前文件所在目录的绝对路径
# current_directory = os.path.dirname(os.path.abspath(__file__))
# 将当前目录添加到sys.path中

print(sys.path)
log = Log("./log/debug.log", "DEBUG").get_logger()
log.info("Starting")
