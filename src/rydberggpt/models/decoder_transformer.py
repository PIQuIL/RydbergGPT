import torch
from torch import nn

from rydberggpt.models.transformer.models import Generator


def get_rydberg_decoder(config):
    output_dim = config.num_states
    n_layers = config.num_blocks
    generator = Generator(config.d_model, config.num_states)

    model = DecoderTransformer(
        input_dim=output_dim,
        d_model=config.d_model,
        n_layers=n_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        generator=generator,
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


class DecoderTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_layers: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        generator: nn.Module,
    ):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Embedding(1000, d_model)
        self.transformer_layers = nn.TransformerEncoderLayer(
            d_model, num_heads, d_ff, dropout, batch_first=True
        )
        self.decoder = nn.TransformerEncoder(self.transformer_layers, n_layers)
        self.dropout = nn.Dropout(dropout)
        self.generator = generator

    def forward(self, tgt: torch.Tensor) -> torch.Tensor:
        """Forward pass for the decoder.

        Args:
            tgt: A tensor of shape (batch_size, seq_len, output_dim) representing the target data.

        Returns:
            A tensor of shape (batch_size, seq_len, d_model) representing the decoded data.
        """
        pos_encoding = self.pos_encoder(
            torch.arange(0, tgt.shape[1], device=tgt.device)
            .unsqueeze(0)
            .expand(tgt.shape[0], -1)
        )
        tgt = self.dropout(self.embedding(tgt) + pos_encoding)
        return self.decoder(tgt)
