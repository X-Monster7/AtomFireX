"""


================
@Author: zhicun Zeng / Alan
@Date: 2023/11/10 11:30
================
"""
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from base.hook_base import HookBase


class Saver(HookBase):
    def __init__(self, save_per_epoch, path = './model_weight/'):
        super().__init__()
        self.save_per_epoch = save_per_epoch
        self.path = path
        os.makedirs(self.path, exist_ok = True)

    def before_running_step(self, index):
        pass

    def after_running_step(self, index):
        if index % self.save_per_epoch == 0:
            torch.save(
                self.trainer.model.state_dict(),
                self.path + f'/epoch_{index}.pth'
            )

    def before_train(self):
        torch.save(
            self.trainer.model.state_dict(),
            self.path + f'origin.pth'
        )

    def after_train(self):
        torch.save(
            self.trainer.model.state_dict(),
            self.path + f'final.pth'
        )


class Writer(HookBase):
    def __init__(self, save_per_epoch, log_dir = './log/tensorboard/'):
        super().__init__()
        self.save_per_epoch = save_per_epoch
        self.writer = SummaryWriter(log_dir = log_dir)

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_running_step(self, index):
        pass

    def after_running_step(self, index):
        """
        自动总结loss，画图
        Returns:

        """
        if index % self.save_per_epoch == 0:
            for key in self.trainer.info['metric']:
                self.writer.add_scalar(key, self.trainer.info['metric'][key], index)


"""
可以继续添加用来测试和评估的部分内容
"""

__all__ = ['Saver', 'Writer']
