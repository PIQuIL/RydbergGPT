# Imports

import torch
from torch import nn

########################################################################################


class WavefunctionBase(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        pass

    def forward(self, x):
        raise NotImplementedError()
        return

    def P(self, x):
        raise NotImplementedError()
        return

    @property
    def psi(self):
        raise NotImplementedError()
        return

    def sample(self, x):
        raise NotImplementedError()
        return

    def varloss(self, H):
        raise NotImplementedError()
        return

    def dataloss(self, x):
        raise NotImplementedError()
        return

    pass
