# Imports

from .wavefunctionbase import WavefunctionBase

import torch

########################################################################################


class TransformerEncoder(torch.nn.Module):
    def __init__(self, N_emb, N_head, N_block, **kwargs):
        super().__init__(**kwargs)
        pass

    def forward(self, x):
        raise NotImplementedError()
        return

    pass


class TransformerDecoder(torch.nn.Module):
    def __init__(self, N_emb, N_head, N_block, **kwargs):
        super().__init__(**kwargs)
        pass

    def forward(self, x):
        raise NotImplementedError()
        return

    pass


########################################################################################


class TransformerWavefunction(WavefunctionBase):
    def __init__(self, N_emb, N_head, N_block, **kwargs):
        super().__init__(**kwargs)

        self.add_module(
            module=TransformerEncoder(N_emb, N_head, N_block), name="encoder"
        )
        self.add_module(
            module=TransformerDecoder(N_emb, N_head, N_block), name="decoder"
        )

        pass

    def forward(self, x):
        raise NotImplementedError()
        return

    def sample(self, x):
        raise NotImplementedError()
        return

    @property
    def psi(self):
        raise NotImplementedError()
        return

    def P_i(self, b):
        raise NotImplementedError()
        return

    def P(self, b):
        raise NotImplementedError()
        return

    def varloss(self, H):
        raise NotImplementedError()
        return

    def dataloss(self, x):
        raise NotImplementedError()
        return

    pass
