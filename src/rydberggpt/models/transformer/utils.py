import copy

import torch.nn as nn


def clones(module: nn.Module, n_clones: int):
    """helper function which produces n_clones copies of a layer"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n_clones)])


# def subsequent_mask(size):
#     """mask out subsequent positions"""
#     attn_shape = (size, size)
#     mask = torch.triu(torch.ones(attn_shape), diagonal=0).type(torch.uint8)
#     # mask = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
#     # mask = mask[0] > mask[1]
#     # return mask
#     return mask == 0


# def make_std_mask(tgt, pad):
#     """Create a mask to hide padding and future words."""
#     tgt_mask = (tgt != pad).unsqueeze(-2)  # noqa
#     tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
#     return tgt_mask
