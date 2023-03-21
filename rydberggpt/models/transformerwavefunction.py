# Imports

from .wavefunctionbase import WavefunctionBase

import torch
from torch import nn

########################################################################################


class TransformerEncoderBlock(nn.Module):
    def __init__(self, N_emb, N_head, **kwargs):
        super().__init__(**kwargs)

        self.add_module(
            module=nn.MultiheadAttention(N_emb, N_head), name="selfattention0"
        )
        self.add_module(module=nn.Linear(N_emb, N_emb), name="linear0")
        pass

    def forward(self, x):
        raise NotImplementedError()
        return

    pass


class TransformerEncoder(nn.Module):
    def __init__(self, N_emb, N_head, N_block, **kwargs):
        super().__init__(**kwargs)

        for i in range(N_block):
            self.add_module(
                module=TransformerEncoderBlock(N_emb, N_head),
                name="encoderblock{}".format(i),
            )

        pass

    def forward(self, x):
        raise NotImplementedError()
        return

    pass


########################################################################################


class TransformerDecoderBlock(nn.Module):
    def __init__(self, N_emb, N_head, **kwargs):
        super().__init__(**kwargs)

        self.add_module(
            module=nn.MultiheadAttention(N_emb, N_head), name="causalattention0"
        )
        self.add_module(
            module=nn.MultiheadAttention(N_emb, N_head), name="encoderdecoderattention0"
        )
        self.add_module(module=nn.Linear(N_emb, N_emb), name="linear0")
        pass

    def forward(self, x):
        raise NotImplementedError()
        return

    pass


class TransformerDecoder(nn.Module):
    def __init__(self, N_emb, N_head, N_block, **kwargs):
        super().__init__(**kwargs)

        for i in range(N_block):
            self.add_module(
                module=TransformerDecoderBlock(N_emb, N_head),
                name="decoderblock{}".format(i),
            )
        pass

    def forward(self, x):
        raise NotImplementedError()
        return

    pass


########################################################################################


class TransformerWavefunction(WavefunctionBase):
    def __init__(self, N_emb=16, N_head=1, N_block=1, **kwargs):
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
