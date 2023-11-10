"""


================
@Author: zhicun Zeng / Alan
@Date: 2023/11/10 11:29
================
"""


class HookBase:
    """
    这是全部hook的基类，基于训练的生命周期设计。
    hook的例子如，tensorboard记录loss、画图；保存模型权重；写文件等等；
    """

    def __init__(self):
        self.trainer = None
        self._info = {}

    def before_train(self):
        """
        Called before the first iteration.
        """
        raise NotImplementedError

    def after_train(self):
        """
        Called after the last iteration.
        """
        raise NotImplementedError

    def before_running_step(self, index):
        """
        Called before each iteration.
        """
        raise NotImplementedError

    def after_running_step(self, index):
        """
        Called after each iteration.
        """
        raise NotImplementedError
