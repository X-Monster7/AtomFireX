"""transformer的封装层
包含seq2seq结构

================
@Author: zhicun Zeng / Alan
@Date: 2023/11/8 20:13
================
"""
from example.net.transformer.Module.index import *


class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.embed = nn.Embedding()
        self.pos_encode = PEN()
        self.multi_attention = Multi_head_attention()
        self.add_norm = Add_norm()

    def forward(self, x):
        pass


class decoder(nn.Module):
    pass


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = encoder()
        self.decoder = decoder()

    def forward(self, x):
        pass


__all__ = ['Transformer']
