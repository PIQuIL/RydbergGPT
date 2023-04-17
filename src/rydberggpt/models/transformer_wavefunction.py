# Imports

import torch
from torch import nn

from .wavefunctionbase import WavefunctionBase

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
        p = x
        p1 = self.selfattention(p, p, p)[0]
        p = p + p1
        p = self.layernorm1(p)
        p1 = self.linear(p)
        p = p + p1
        p = self.layernorm2(p)
        return p

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
        p = x

        for i in range(self.N_block):
            p = self._modules["encoderblock{}".format(i)](p)

        return p

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
        p, y = x

        N_seq = y.shape[-2]
        c = torch.meshgrid(torch.arange(N_seq), torch.arange(N_seq), indexing="ij")
        c = c[0] > c[1]

        y1 = self.causalattention(y, y, y, attn_mask=c)[0]
        y = y + y1
        y = self.layernorm1(y)
        y1 = self.encoderdecoderattention(y, p, p)[0]
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
        p, y = x

        for i in range(self.N_block):
            y = self._modules["decoderblock{}".format(i)](x)

        return y

    pass


########################################################################################


class TransformerWavefunction(WavefunctionBase):
    def __init__(self, N_emb=16, N_head=1, N_block=1, **kwargs):
        super().__init__(**kwargs)

        self.add_module(module=nn.Linear(4, N_emb), name="encoderembedding")
        self.add_module(module=nn.Linear(2, N_emb), name="decoderembedding")
        self.add_module(
            module=TransformerEncoder(N_emb, N_head, N_block), name="encoder"
        )
        self.add_module(
            module=TransformerDecoder(N_emb, N_head, N_block), name="decoder"
        )
        self.add_module(module=nn.Linear(N_emb, 2), name="linear")
        self.add_module(module=nn.Softmax(-1), name="softmax")

        pass

    def forward(self, x):
        p, y = x

        # y = nn.functional.one_hot(y, 2)
        # y = y.to(torch.float)

        p = self.encoderembedding(p)
        y = self.decoderembedding(y)

        p = self.encoder(p)
        y = self.decoder([p, y])

        y = self.linear(y)
        y = self.softmax(y)

        return y

    def P(self, x):
        p, y = x

        y = nn.functional.one_hot(y, 2)
        y = y.to(torch.float)

        P = torch.prod(torch.sum(self(x) * y, axis=-1), axis=-1)
        return P

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


########################################################################################
