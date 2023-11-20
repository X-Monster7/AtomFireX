"""


================
@Author: zhicun Zeng / Alan
@Date: 2023/11/19 11:51
================
"""
from example.net.transformer.module.core import *


class EncoderBlock(nn.Module):
    """
    单一的transformer encoder block.
    """

    def __init__(self, normed_shape, dropout, qkv_size, h, ffn_hidden):
        super(EncoderBlock, self).__init__()
        self.add_norm1 = AddNorm(normed_shape, dropout)
        self.multi_attention = MultiHeadAttention(qkv_size, h)
        self.ffn = PositionWiseFFN(qkv_size, ffn_hidden)
        self.add_norm2 = AddNorm(normed_shape, dropout)

    def forward(self, x, y, z):
        x_ = x
        x = self.multi_attention(x, y, z)
        x = self.add_norm1(x_, x)
        x_ = x
        x = self.ffn(x)
        x = self.add_norm2(x_, x)
        return x
