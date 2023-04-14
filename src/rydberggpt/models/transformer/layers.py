from typing import List, Tuple

import torch
import torch.nn as nn

from rydberggpt.models.transformer.modules import (
    PositionwiseFeedForward,
    SublayerConnection,
)
from rydberggpt.models.transformer.utils import clones


class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward.

    Parameters:
        size (int): The input size.
        self_attn (nn.MultiheadAttention): The self-attention module.
        src_attn (nn.MultiheadAttention): The source-attention module.
        feed_forward (PositionwiseFeedForward): The feed forward module.
        dropout (float): The dropout rate.
    """

    def __init__(
        self,
        size: int,
        self_attn: nn.MultiheadAttention,
        src_attn: nn.MultiheadAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float,
    ):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass through the decoder.

        Parameters:
            x (torch.Tensor): The input tensor.
            memory (torch.Tensor): The memory tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, is_causal=True)[0])
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m)[0])
        return self.sublayer[2](x, self.feed_forward)


class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed forward.

    Parameters:
        size (int): The input size.
        self_attn (nn.MultiheadAttention): The self-attention module.
        feed_forward (PositionwiseFeedForward): The feed forward module.
        dropout (float): The dropout rate.
    """

    def __init__(
        self,
        size: int,
        self_attn: nn.MultiheadAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float,
    ):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass through the encoder.

        Parameters:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x)[0])
        return self.sublayer[1](x, self.feed_forward)


#########
# alternative implementation
class DecoderLayer1(nn.Module):
    def __init__(self, d_model, num_heads, config, **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.num_heads = num_heads
        self.config = config

        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.src_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ln = clones(nn.LayerNorm(d_model), 3)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x, memory):
        x = x + self.self_attn(x, x, x, is_causal=True)[0]
        x = x + self.src_attn(self.ln[0](x), memory, memory)[0]
        x = x + self.linear(self.ln[1](x))
        return self.ln[2](x)


# atlernative implementation, this layer converges a bit faster
class EncoderLayer1(nn.Module):
    def __init__(self, d_model, num_heads, config, **kwargs):
        super().__init__(**kwargs)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ln = clones(nn.LayerNorm(d_model), 2)
        self.linear = nn.Linear(d_model, d_model)
        # self.config = config

    def forward(self, x):
        x = x + self.attn(x, x, x)[0]
        x = x + self.linear(self.ln[0](x))
        return self.ln[1](x)
