# -*- coding: utf-8 -*-

"""
@author: Alan Tsang / Zeng Zhicun
@institution: CSU, China, changsha
@date: 2023/9/21 16:26
@version: 1.0.0
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 100)
y = np.random.rand(100)
y_2 = np.arange(0, 100)
y_3 = np.arange(0, 100) + y

def _subplot():
    # plt.figure(1)
    # plt.subplot(311)
    # plt.plot(x, y)
    #
    # plt.subplot(312)
    # plt.plot(x, y_2)
    #
    # plt.subplot(313)
    # plt.plot(x, y_3)

    # 设置图形大小
    plt.figure(figsize=(8, 6))
    for i, data in enumerate([y, y_2, y_3], start=1):
        # plt.figure(i): one figure, one window
        plt.subplot(3, 1, i)
        plt.plot(x, data)

    # 调整子图之间的间距
    plt.tight_layout()
    # show will delete the
    plt.savefig('./img/subplot.png')

    plt.show()

def _subplots():
    # plt.figure(1)
    # plt.subplot(311)
    # plt.plot(x, y)
    #
    # plt.subplot(312)
    # plt.plot(x, y_2)
    #
    # plt.subplot(313)
    # plt.plot(x, y_3)

    # 设置图形大小
    plt.figure(figsize=(8, 6))
    for i, data in enumerate([y, y_2, y_3], start=1):
        # plt.figure(i): one figure, one window
        plt.subplot(3, 1, i)
        plt.plot(x, data)

    # 调整子图之间的间距
    plt.tight_layout()
    # show will delete the
    plt.savefig('./img/subplot.png')

    plt.show()

_subplot()




