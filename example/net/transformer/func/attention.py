"""基础的点积注意力机制实现


================
@Author: zhicun Zeng / Alan
@Date: 2023/11/8 20:33
================
"""

import numpy as np
import torch
import torch.nn.functional as F
from aspect.general import timer, Log

Tensor = torch.Tensor


@timer
# @Log
def dot_production_attention(q: Tensor, k: Tensor, v: Tensor, valid_len = None) -> Tensor:
    """
    点积注意力
    Args:
        valid_len: valid length for k、v
        q: query, (batch_size, query_num, d)
        k: key, (batch_size, key_value_pair_num, d)
        v: value, (batch_size, ) or (batch_size, 查询个数)

    Returns:
       (batch_size, query_num, key_value_pair_num)
    """
    d_k = q.shape[-1]
    _ = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(d_k)
    if valid_len is not None:
        _ = masked_softmax(_, valid_len)
    else:
        # dim指定为1，在每个查询中计算其与所有键的相关性得分，
        # 然后使用 softmax 将这些分数归一化为注意力权重
        _ = F.softmax(_, dim = -1)
        # question: why _ can bmm with v ?
    return torch.bmm(_, v)


def masked_softmax(X: Tensor, valid_lens: Tensor):
    """
    在最后一个轴上掩蔽元素来执行softmax操作.
    X的前面两个维度直接合起来就可以了
    Args:
        X: (batch, len(text), hidden(query_size))
        valid_lens: (mask_length)每一个batch的text mask长度 or
        (len(text) , mask_length) 不同的batch的text长度

    Returns:

    """
    """通过"""
    # X:3D张量，valid_lens:1D或2D张量
    shape = X.shape
    if valid_lens.dim() == 1:
        valid_lens = torch.repeat_interleave(valid_lens, shape[1])
    else:
        valid_lens = valid_lens.reshape(-1)
    # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
    X = sequence_mask(
        X.reshape(-1, shape[-1]), valid_lens,
        value = -1e6
    )
    return F.softmax(X.reshape(shape), dim = -1)


def sequence_mask(X, valid_len, value: float = 0):
    maxlen = X.size(1)
    # arange (1, max_len) 与 valid_len (, 1) 都自动广播了]
    mask = torch.arange(maxlen, dtype = torch.float32, device = X.device)[None, :] < valid_len[:, None]
    # X 和 mask的shape相同
    X[~mask] = value
    return X


def transpose_qkv(x: torch.Tensor, h: int) -> Tensor:
    """
    分离x中的hidden为h份，然后将h维度并入shape[0]中
    tip：可以通过增加维度的方式，将数据x(q, k, v)拆分，而不仅限于拆分为多个列表
    Args:
        x: (batch_size, 查询数num, hidden)
        h: int

    Returns:
        (batch_size * num_heads, 查询或者“键－值”对的个数, num_hidden / num_heads)
    """
    x = x.reshape(x.shape[0], x.shape[1], h, -1)
    x = x.transpose(2, 1)
    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hidden / num_heads)
    return x.reshape(-1, x.shape[2], x.shape[3])


def untranspose_qkv(x: Tensor, h: int) -> Tensor:
    """
    将x中合并的第一维度（batch_size * h）分离为 batch_size, h 两个维度，然后 将h并入到最后一列的hidden中
    Args:
        x: (batch_size * h, num_query / num_key_value , hidden)
        h: int

    Returns:
        (batch_size,  num_query / num_key_value, h * hidden)
    """
    x = x.reshape(-1, h, x.shape[1], x.shape[2])
    x = x.transpose(2, 1)
    return x.reshape(x.shape[0], x.shape[1], -1)


if __name__ == '__main__':
    print(masked_softmax(torch.rand(2, 3, 4), torch.tensor([2, 3])))
