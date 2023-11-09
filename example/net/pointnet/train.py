#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

@description:

==========================================
@author: Alan Tsang / Zeng Zhicun
@institution: CSU, China, changsha
@date: 2023/10/23 9:02
@version: 1.0.0
"""
import sys
import os
from example.net import pointnet
import torch

sys.path.append(f'../../example')

# 创建模型
model = pointnet(1024)
model.train()
# 优化器定义
optim = torch.optim.Adam(params = model.parameters(), weight_decay = 0.001)
# 损失函数定义
loss_fn = torch.nn.CrossEntropyLoss()
# 评价指标定义
m = torch.metric.Accuracy()
# 训练轮数
epoch_num = 50
# 每多少个epoch保存
save_interval = 2
# 每多少个epoch验证
val_interval = 2
best_acc = 0
# 模型保存地址
output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# 训练过程
plot_acc = []
plot_loss = []
for epoch in range(epoch_num):
    total_loss = 0
    for batch_id, data in enumerate(train_loader()):
        inputs = torch.to_tensor(data[0], dtype = 'float32')
        labels = torch.to_tensor(data[1], dtype = 'int64')
        predicts = model(inputs)

        # 计算损失和反向传播
        loss = loss_fn(predicts, labels)
        total_loss = total_loss + loss.numpy()[0]
        loss.backward()
        # 计算acc
        predicts = torch.reshape(predicts, (predicts.shape[0] * predicts.shape[1], -1))
        labels = torch.reshape(labels, (labels.shape[0] * labels.shape[1], 1))
        correct = m.compute(predicts, labels)
        m.update(correct)
        # 优化器更新
        optim.step()
        optim.clear_grad()
    avg_loss = total_loss / batch_id
    plot_loss.append(avg_loss)
    print("epoch: {}/{}, loss is: {}, acc is:{}".format(epoch, epoch_num, avg_loss, m.accumulate()))
    m.reset()
    # 保存
    if epoch % save_interval == 0:
        model_name = str(epoch)
        torch.save(model.state_dict(), './output/PointNet_{}.pdparams'.format(model_name))
        torch.save(optim.state_dict(), './output/PointNet_{}.pdopt'.format(model_name))
    # 训练中途验证
    if epoch % val_interval == 0:
        model.eval()
        for batch_id, data in enumerate(val_loader()):
            inputs = torch.to_tensor(data[0], dtype = 'float32')
            labels = torch.to_tensor(data[1], dtype = 'int64')
            predicts = model(inputs)
            predicts = torch.reshape(predicts, (predicts.shape[0] * predicts.shape[1], -1))
            labels = torch.reshape(labels, (labels.shape[0] * labels.shape[1], 1))
            correct = m.compute(predicts, labels)
            m.update(correct)
        val_acc = m.accumulate()
        plot_acc.append(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            print("===================================val===========================================")
            print('val best epoch in:{}, best acc:{}'.format(epoch, best_acc))
            print("===================================train===========================================")
            torch.save(model.state_dict(), './output/best_model.pdparams')
            torch.save(optim.state_dict(), './output/best_model.pdopt')
        m.reset()
        model.train()
