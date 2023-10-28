#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: Alan Tsang / Zeng Zhicun
@institution: CSU, China, changsha
@date: 2023/10/18 22:20
@version: 1.0.0
"""

import os
import sys

from util.logger_util.index import Log

# # 获取当前文件所在目录的绝对路径
# current_directory = os.path.dirname(os.path.abspath(__file__))
# # 将当前目录添加到sys.path中
# sys.path.append(current_directory)
log = Log("./log/log", "DEBUG").get_logger()
log.info("Starting")
