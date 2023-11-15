"""


==================================
Author: Alan / Zeng Zhicun
Institution: CSU, China, changsha
Date: 2023/10/28
==================================
"""

import time
import os
import logging
import functools
from functools import wraps
import pprint
import types


def timer(func):
    """
    装饰器用法，将函数的运算时间打印到控制台
    Args:
        func ():

    Returns:

    """

    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__}, id {id(func)}, 用时 {end - start} s")
        return res

    return wrapper


class Log:
    # TODO: 测试@ 传参、测试能够正常用于函数和类内部的函数。
    """
    日志记录类，类装饰器用法（请参见AtomFire/reference/class_decorator/index.py文件）
    """
    LEVEL = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    def __init__(self, func, path = "./Log/.log", level = "DEBUG"):
        wraps(func)(self)
        base_path = os.path.split(path)[0]
        os.makedirs(base_path, exist_ok = True)
        logging.basicConfig(
            filename = path,
            level = Log.LEVEL[level],
            format = "%(asctime)s [%(levelname)s] %(message)s",
            datefmt = "%Y-%m-%d %H:%M:%S",
        )
        self.logger = logging.getLogger("logger")

    def __call__(self, func):
        @functools.wraps
        def wrapper(*args, **kwargs):
            try:
                res = self.__wrapped__(args, kwargs)
                # TODO: 添加更加合理、有效的参数信息
                self.logger.info(f"{func.__name__} 执行成功")
            except BaseException:
                res = None
                # TODO: 添加更加合理、有效的参数信息
                self.logger.error(f"{func.__name__} 执行成功")
            return res

        return wrapper

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return types.MethodType(self, instance)
