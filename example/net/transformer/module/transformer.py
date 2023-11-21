"""transformer的封装层
包含seq2seq结构

================
@Author: zhicun Zeng / Alan
@Date: 2023/11/8 20:13
================
"""
import yaml
import math
import sys

import torch

from example.net.transformer.module.block import *
from example.net.transformer.module.core import *


class TransformerEncoder(nn.Module):
    """编码器"""
    """
    valid_len是1D或者2D，最终变成长度为（b * h, len(text) ）的一维向量
    valid是1D，考虑多头的话，valid_len 需要 先在多头注意力层 0维放大h倍，
    然后在mask_softmax层放大第1维。
    
    valid是2D，自身就考虑了第1维，因此只在多头注意力层 0维放大h倍。
    """

    def __init__(self, args: dict):
        super().__init__()
        train_args = args['model']['encoder']
        h = args['model']['h']
        # key_size = train_args['key_size']
        # query_size = train_args['query_size']
        # value_size = train_args['value_size']
        text_hidden = train_args['text_hidden']
        text_size = train_args['text_size']
        layer = train_args['layer']
        dropout = train_args['dropout']
        ffn_hidden = train_args['ffn_hidden']
        self.text_hidden = train_args['text_hidden']
        self.encoder_list = nn.ModuleList(
            [
                EncoderBlock(
                    normed_shape = [text_size, text_hidden],
                    dropout = dropout,
                    qkv_size = text_hidden,
                    h = h,
                    ffn_hidden = ffn_hidden
                )
                for _ in range(layer)
            ]
        )
        self.embedding = nn.Embedding(text_size, text_hidden)
        self.pos_encode = PEN(text_hidden)

    def forward(self, x, valid_len):
        x = self.embedding(x) * math.sqrt(self.text_hidden)
        x = self.pos_encode(x)
        for encoder in self.encoder_list:
            x = encoder(x, x, x, valid_len)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, args: dict):
        super().__init__()
        train_args = args['model']['encoder']
        h = args['model']['h']
        # key_size = train_args['key_size']
        # query_size = train_args['query_size']
        # value_size = train_args['value_size']
        text_hidden = train_args['text_hidden']
        text_size = train_args['text_size']
        layer = train_args['layer']
        dropout = train_args['dropout']
        ffn_hidden = train_args['ffn_hidden']


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = -666
        self.decoder = TransformerDecoder()

    def forward(self, x):
        pass


__all__ = ['Transformer']

if __name__ == '__main__':
    with open('../config/index.yml') as _:
        args = yaml.safe_load(_)
    encoder = TransformerEncoder(args)
    encoder.eval()
    valid_len = torch.tensor([3, 2])
    print(encoder(torch.ones((2, 100), dtype = torch.long), valid_len).shape)
