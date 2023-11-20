"""


================
@Author: zhicun Zeng / Alan
@Date: 2023/11/10 11:38
================
"""
from tqdm import tqdm
from base.trainer_base import TrainerBase


class Trainer(TrainerBase):
    """
    使用流程：
    1. 创建一个Trainer，指定data_loader, model, loss, optimizer, config.
       config拟采用yaml加载的超参数。
    2. 挂载hooks，register函数
    3. 重写run的逻辑
    4. 运行fit函数
    """

    def __init__(self, data_loader, model, loss, optimizer, config):
        super().__init__()

        # 注意到为了灵活性，这里仍然没有定义 data_loader , model 和 optimizer
        # 仍然是采用了 加载的方式，而真正定义这些的类，会在下一节中介绍
        model.train()
        self.data_loader = data_loader
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        # 用yaml包加载配置，yaml加载之后是一个字典
        self.config = config
        self.info = {'metric': {}}
        for _type in self.config['metric']:
            self.info['metric'][_type] = None

    def fit(self):
        try:
            self.before_train()
            for index in tqdm(range(self.config['train']['epoch'])):
                self.before_running_step(index)
                self._run()
                self.after_running_step(index)
        finally:
            self.after_train()

    def _run(self):
        for (data, Y) in self.data_loader:
            data = data.cuda()
            Y = Y.cuda()
            y = self.model(data)
            loss = self.loss(y, Y)
            self.info['metric']['loss'] = loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
