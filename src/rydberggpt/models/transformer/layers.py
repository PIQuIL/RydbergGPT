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

    Args:
        size (int): The input size. (d_model)
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

    def forward(
        self, x: torch.Tensor, memory: torch.Tensor, batch_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the forward pass through the decoder.

        Args:
            x (torch.Tensor): The input tensor.
            memory (torch.Tensor): The memory tensor.
            batch_mask (torch.Tensor): The mask tensor for batches.

        Returns:
            torch.Tensor: The output tensor.
        """

        causal_attn_mask = torch.meshgrid(
            torch.arange(x.shape[-2], device=x.device),
            torch.arange(x.shape[-2], device=x.device),
            indexing="ij",
        )
        causal_attn_mask = causal_attn_mask[0] >= causal_attn_mask[1]
        causal_attn_mask = torch.logical_not(causal_attn_mask)

        batch_key_mask = batch_mask
        batch_key_mask = torch.logical_not(batch_key_mask)

        m = memory
        x = self.sublayer[0](
            x, lambda x: self.self_attn(x, x, x, attn_mask=causal_attn_mask)[0]
        )
        x = self.sublayer[1](
            x, lambda x: self.src_attn(x, m, m, key_padding_mask=batch_key_mask)[0]
        )
        return self.sublayer[2](x, self.feed_forward)


class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed forward.

    Args:
        size (int): The input size. (d_model)
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

    def forward(self, x: torch.Tensor, batch_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass through the encoder.

        Args:
            x (torch.Tensor): The input tensor.
            batch_mask (torch.Tensor): The mask tensor for batches.

        Returns:
            torch.Tensor: The output tensor.
        """

        batch_key_mask = batch_mask
        batch_key_mask = torch.logical_not(batch_key_mask)

        x = self.sublayer[0](
            x,
            lambda x: torch.nan_to_num(
                self.self_attn(x, x, x, key_padding_mask=batch_key_mask)[0]
            ),
        )
        return self.sublayer[1](x, self.feed_forward)
