"""


================
@Author: zhicun Zeng / Alan
@Date: 2023/11/10 11:35
================
"""
import weakref

from example.train_frame.base.hook_base import HookBase


class TrainerBase:

    def __init__(self):
        self._hooks = []

    def fit(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def register_hooks(self, hooks):
        for h in hooks:
            # assert (h is not None) and isinstance(h, HookBase)
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def unregister_hooks(self):
        self._hooks = []
        return True

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        for h in self._hooks:
            h.after_train()

    def before_running_step(self, index):
        for h in self._hooks:
            h.before_running_step(index)

    def after_running_step(self, index):
        for h in self._hooks:
            h.after_running_step(index)
