#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: Alan Tsang / Zeng Zhicun
@institution: CSU, China, changsha
@date: 2023/10/18 22:03
@version: 1.0.0
"""

import logging
import os

LEVEL = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING,
         "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}


class Log:
    def __init__(self, path, level):
        base_path = os.path.split(path)[0]
        os.makedirs(base_path, exist_ok = True)
        logging.basicConfig(
            filename = path,
            level = LEVEL[level],
            format = '%(asctime)s [%(levelname)s] %(message)s',
            datefmt = '%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger('logger')

    def get_logger(self, *args, **kwargs):
        return self.logger
