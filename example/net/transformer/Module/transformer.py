"""transformer的封装层
包含seq2seq结构

================
@Author: zhicun Zeng / Alan
@Date: 2023/11/8 20:13
================
"""
import math
import sys

import torch

sys.path.append('G:\\AtomFire\\')

from example.net.transformer.Module.index import *


class EncoderBlock(nn.Module):
    """
    单一的transformer encoder block.
    """

    def __init__(self, key_size, query_size, value_size, normed_shape, dropout,
                 in_channel, hidden_channel, out_channel):
        super(EncoderBlock, self).__init__()
        self.add_norm1 = AddNorm(normed_shape, dropout)
        self.multi_attention = MultiHeadAttention(query_size, key_size, value_size, 24, 2, 24)
        self.ffn = PositionWiseFFN(in_channel, hidden_channel, out_channel)
        self.add_norm2 = AddNorm(normed_shape, dropout)

    def forward(self, x, y, z):
        x_ = x
        x = self.multi_attention(x, y, z)
        x = self.add_norm1(x_, x)
        x_ = x
        x = self.ffn(x)
        x = self.add_norm2(x_, x)
        return x


class TransformerEncoder(nn.Module):
    """编码器"""

    def __init__(self, layer, vocab_size, num_hidden, key_size, query_size, value_size, normed_shape, dropout,
                 in_channel, hidden_channel, out_channel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_list = nn.ModuleList(
            [EncoderBlock(
                key_size, query_size, value_size, normed_shape, dropout, in_channel, hidden_channel, out_channel
            ) for i in range(layer)]
        )
        self.num_hidden = num_hidden
        self.embedding = nn.Embedding(vocab_size, num_hidden)
        self.pos_encode = PEN(num_hidden)

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.num_hidden)
        x = self.pos_encode(x)
        for encoder in self.encoder_list:
            x = encoder(x, x, x)
        return x


class decoder(nn.Module):
    pass


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = ()
        self.decoder = decoder()

    def forward(self, x):
        pass


__all__ = ['Transformer']

if __name__ == '__main__':
    # batch_size, len(seq), query_size(注释：词向量的维度)
    # X = torch.ones((2, 100, 24))
    # Y = torch.ones((2, 100, 24))
    # Z = torch.ones((2, 100, 24))
    # encoder_blk = EncoderBlock(
    #     24, 24, 24,
    #     [100, 24], 0.3, 24, 24, 24
    # )
    # X = X.cuda()
    # Y = Y.cuda()
    # Z = Z.cuda()
    # encoder_blk = encoder_blk.cuda()
    #
    # encoder_blk.eval()
    # print(encoder_blk(X, Y, Z).shape)
    encoder = TransformerEncoder(
        2, 100, 24, 24, 24, 24, [100, 24], 0.5, 24, 24, 24
    )
    encoder.eval()
    print(encoder(torch.ones((2, 100), dtype = torch.long)).shape)
