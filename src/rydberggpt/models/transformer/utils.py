import copy

import torch.nn as nn


def clones(module: nn.Module, n_clones: int):
    """helper function which produces n_clones copies of a layer"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n_clones)])
