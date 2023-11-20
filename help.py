"""


================
@Author: zhicun Zeng / Alan
@Date: 2023/11/17 19:59
================
"""
import torch
import matplotlib.pyplot as plt

n = 100
x_ = []
y_ = []
for N in range(1, n + 1):
    x = torch.rand(N, requires_grad = True)
    l1 = torch.log(1 / N * torch.sum(x))

    y = torch.rand(N, requires_grad = True)
    l2 = 1 / N * torch.sum(torch.log(y))

    l1.backward()

    l2.backward()

    x_.append(torch.sum(x.grad))
    y_.append(torch.sum(y.grad))

plt.plot(range(n), x_)
plt.show()
plt.plot(range(n), y_)
plt.show()
