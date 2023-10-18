import copy

import torch
import torch.nn as nn


########################################################################################


def clones(module: nn.Module, n_clones: int):
    """helper function which produces n_clones copies of a layer"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n_clones)])


########################################################################################


def snake_flip(x: torch.Tensor) -> torch.Tensor:
    """
    Implements a "snake" flip which reorders the 2D tensor into snake order when flattened.

    Args:
        x (torch.Tensor): The tensor to apply the snake flip to, dimensions should be [..., Ly, Lx].

    Returns:
        torch.Tensor: The "snake" flipped tensor, dimensions will be [..., Ly, Lx].
    """
    if isinstance(x, torch.Tensor):
        raise TypeError("Function only supports torch.Tensor")

    _x = copy.deepcopy(x)

    for i in range(_x.shape[-2]):
        if i % 2 == 1:
            _x[..., i, :] = torch.flip(_x[..., i, :], dims=(-1,))

    return _x


def flattened_snake_flip(x: torch.Tensor, Lx: int, Ly: int) -> torch.Tensor:
    """
    Implements a "snake" flip which reorders the flattened 2D tensor into snake order.

    Args:
        x (torch.Tensor): The tensor to apply the snake flip to, dimensions should be [..., Ly * Lx].

    Returns:
        torch.Tensor: The "snake" flipped tensor, dimensions will be [..., Ly * Lx].
    """
    return snake_flip(x.reshape(*x.shape[:-1], Ly, Lx)).reshape(*x.shape[:-1], -1)
