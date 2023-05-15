import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rydberggpt.models.transformer.utils import clones


class SublayerConnection(nn.Module):
    """
    This module implements a residual connection followed by a layer norm.

    Args:
        size (int): The input size.
        dropout (float): The dropout rate.
    """

    def __init__(self, size: int, dropout: float):
        super(SublayerConnection, self).__init__()
        self.layer_norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """
        Compute the forward pass through the module.

        Args:
            x (torch.Tensor): The input tensor.
            sublayer (nn.Module): The sublayer module.

        Returns:
            torch.Tensor: The output tensor.
        """
        # NOTE For GPT2 the authors moved Layer normalization (Ba et al., 2016)
        # to the input of each sub-block.
        # see Sec. 2.3 https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
        return x + self.dropout(sublayer(self.layer_norm(x)))
        # return self.layer_norm(x + self.dropout(sublayer(x)))


class PositionwiseFeedForward(nn.Module):
    """
    A two-layer feed-forward network.

    Args:
        d_model (int): The input size.
        d_ff (int): The hidden size.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass through the module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    """
    The embedding layer.

    Args:
        d_model (int): The embedding size.
        vocab_size (int): The vocabulary size.
    """

    def __init__(self, d_model: int, vocab_size: int):
        super(Embeddings, self).__init__()
        self.lut = nn.Linear(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass through the module.

        Parameters:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.lut(x) * math.sqrt(self.d_model)
        return x


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding2D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # compute the positional encodings once in log space
        assert d_model % 2 == 0, "d_model must be even for 2D positional encoding"
        pe = torch.zeros(max_len, d_model // 2)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x, dim_1: int, dim_2: int):
        # Compute the positional encodings along two dimensions, then concatenate them
        N, D = x.size(0), x.size(1)
        assert N == dim_1 * dim_2
        pe_1 = self.pe[:dim_1, :D]
        pe_2 = self.pe[:dim_2, :D]
        pe_1 = pe_1.unsqueeze(1).repeat(1, dim_2, 1).view(-1, D)
        pe_2 = pe_2.unsqueeze(0).repeat(dim_1, 1, 1).view(-1, D)
        pe = torch.cat([pe_1, pe_2], dim=1)
        out = x + pe
        return self.dropout(out)
