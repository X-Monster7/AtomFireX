#!/usr/bin/config python
# -*- coding: utf-8 -*-

"""
@author: Alan Tsang / Zeng Zhicun
@institution: CSU, China, changsha
@date: 2023/10/17 14:09
@version: 1.0.0
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 生成虚拟的点云数据
num_points = 1000
x = np.random.rand(num_points)  # 随机生成x坐标
y = np.random.rand(num_points)  # 随机生成y坐标
z = 0.1 * x + 0.2 * y + np.random.normal(0, 0.02, num_points)  # 生成z坐标，带一些噪声

# 创建一个3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

# 绘制点云
ax.scatter(x, y, z, c = 'b', marker = 'o')

# 设置坐标轴标签
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')
ax.set_zlabel('Z轴')

plt.savefig('./img/point_cloud.png')
# 显示图形
plt.show()
