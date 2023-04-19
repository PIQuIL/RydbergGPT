from typing import Tuple

import torch
import torch.nn as nn

from rydberggpt.models.transformer.models import Generator


def get_rydberg_encoder_decoder_2(config):
    input_dim = config.num_encoder_embedding_dims
    output_dim = config.num_states
    n_layers = config.num_blocks

    encoder = Encoder(
        input_dim,
        config.d_model,
        n_layers,
        config.num_heads,
        config.d_ff,
        config.dropout,
    )
    decoder = Decoder(
        output_dim,
        config.d_model,
        n_layers,
        config.num_heads,
        config.d_ff,
        config.dropout,
    )

    generator = Generator(config.d_model, 2)
    model = EncoderDecoder(encoder, decoder, generator)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_layers: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Embedding(1000, d_model)
        self.transformer_layers = nn.TransformerEncoderLayer(
            d_model, num_heads, d_ff, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.transformer_layers, n_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """Forward pass for the encoder.

        Args:
            src: A tensor of shape (batch_size, input_dim) representing the input data.

        Returns:
            A tensor of shape (batch_size, seq_len, d_model) representing the encoded data.
        """
        pos_embedding = self.pos_encoder(torch.arange(0, src.shape[0]).unsqueeze(1))
        # NOTE The positional encoding for the encoder is not needed.
        src = self.dropout(self.embedding(src) + pos_embedding)
        return self.encoder(src)


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        d_model: int,
        n_layers: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding = nn.Linear(output_dim, d_model)
        self.pos_encoder = nn.Embedding(1000, d_model)
        self.transformer_layers = nn.TransformerDecoderLayer(
            d_model, num_heads, d_ff, dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.transformer_layers, n_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        """Forward pass for the decoder.

        Args:
            tgt: A tensor of shape (batch_size, output_dim) representing the target data.
            encoder_output: A tensor of shape (batch_size, seq_len, d_model) representing the output from the encoder.

        Returns:
            A tensor of shape (batch_size, seq_len, d_model) representing the decoded data.
        """
        pos_embedding = self.pos_encoder(torch.arange(0, tgt.shape[0]).unsqueeze(1))
        tgt = self.dropout(self.embedding(tgt) + pos_embedding)
        return self.decoder(tgt, encoder_output)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, generator: Generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Forward pass for the encoder-decoder architecture.

        Args:
            src: A tensor of shape (batch_size, input_dim) representing the input data.
            tgt: A tensor of shape (batch_size, output_dim) representing the target data.

        Returns:
            A tensor of shape (batch_size, seq_len, d_model) representing the output of the decoder.
        """
        return self.decoder(tgt, self.encoder(src))
