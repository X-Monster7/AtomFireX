"""第一个Cython入门代码


================
@Author: zhicun Zeng / Alan
@Date: 2023/11/8 12:57
================
"""

from distutils.core import setup
from Cython.Build import cythonize

# 没有安装VS C++的环境，无法运行
setup(name = 'helloWorld', ext_modules = cythonize('hello_world.pyx'))

