# Imports

from .wavefunctionbase import WavefunctionBase

import torch
from torch import nn

########################################################################################


class TransformerEncoderBlock(nn.Module):
    def __init__(self, N_emb, N_head, **kwargs):
        super().__init__(**kwargs)

        self.N_emb = N_emb
        self.N_head = N_head

        self.add_module(
            module=nn.MultiheadAttention(N_emb, N_head, batch_first=True),
            name="selfattention",
        )
        self.add_module(module=nn.LayerNorm(N_emb), name="layernorm1")
        self.add_module(module=nn.Linear(N_emb, N_emb), name="linear")
        self.add_module(module=nn.LayerNorm(N_emb), name="layernorm2")
        pass

    def forward(self, x):
        y = x
        y1 = self.selfattention(y, y, y)[0]
        y = y + y1
        y = self.layernorm1(y)
        y1 = self.linear(y)
        y = y + y1
        y = self.layernorm2(y)
        return y

    pass


class TransformerEncoder(nn.Module):
    def __init__(self, N_emb, N_head, N_block, **kwargs):
        super().__init__(**kwargs)

        self.N_emb = N_emb
        self.N_head = N_head
        self.N_block = N_block

        for i in range(N_block):
            self.add_module(
                module=TransformerEncoderBlock(N_emb, N_head),
                name="encoderblock{}".format(i),
            )

        pass

    def forward(self, x):
        y = x

        for i in range(self.N_block):
            y = self._modules["encoderblock{}".format(i)](y)

        return y

    pass


########################################################################################


class TransformerDecoderBlock(nn.Module):
    def __init__(self, N_emb, N_head, **kwargs):
        super().__init__(**kwargs)

        self.N_emb = N_emb
        self.N_head = N_head

        self.add_module(
            module=nn.MultiheadAttention(N_emb, N_head, batch_first=True),
            name="causalattention",
        )
        self.add_module(module=nn.LayerNorm(N_emb), name="layernorm1")
        self.add_module(
            module=nn.MultiheadAttention(N_emb, N_head, batch_first=True),
            name="encoderdecoderattention",
        )
        self.add_module(module=nn.LayerNorm(N_emb), name="layernorm2")
        self.add_module(module=nn.Linear(N_emb, N_emb), name="linear")
        self.add_module(module=nn.LayerNorm(N_emb), name="layernorm3")
        pass

    def forward(self, x):
        y, z = x

        N_seq = y.shape[-2]
        c = torch.meshgrid(torch.arange(N_seq), torch.arange(N_seq), indexing="ij")
        c = c[0] > c[1]

        y1 = self.causalattention(y, y, y, attn_mask=c)[0]
        y = y + y1
        y = self.layernorm1(y)
        y1 = self.encoderdecoderattention(y, z, z)[0]
        y = y + y1
        y = self.layernorm2(y)
        y1 = self.linear(y)
        y = y + y1
        y = self.layernorm3(y)
        return y

    pass


class TransformerDecoder(nn.Module):
    def __init__(self, N_emb, N_head, N_block, **kwargs):
        super().__init__(**kwargs)

        self.N_emb = N_emb
        self.N_head = N_head
        self.N_block = N_block

        for i in range(N_block):
            self.add_module(
                module=TransformerDecoderBlock(N_emb, N_head),
                name="decoderblock{}".format(i),
            )
        pass

    def forward(self, x):
        y = x

        for i in range(self.N_block):
            y = self._modules["decoderblock{}".format(i)](y)

        return y

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
