import copy
from typing import Tuple

import torch
from torch import nn

from rydberggpt.models.transformer.layers import DecoderLayer, EncoderLayer
from rydberggpt.models.transformer.models import (
    Decoder,
    Encoder,
    EncoderDecoder,
    Generator,
)
from rydberggpt.models.transformer.modules import PositionwiseFeedForward


def get_rydberg_transformer(config):
    c = copy.deepcopy
    attn = nn.MultiheadAttention(config.d_model, config.num_heads, batch_first=True)
    ff = PositionwiseFeedForward(config.d_model, config.d_ff, config.dropout)

    model = RydbergTransformer(
        encoder=Encoder(
            EncoderLayer(config.d_model, c(attn), c(ff), config.dropout),
            config.num_blocks,
        ),
        decoder=Decoder(
            DecoderLayer(config.d_model, c(attn), c(attn), c(ff), config.dropout),
            config.num_blocks,
        ),
        src_embed=nn.Linear(config.num_encoder_embedding_dims, config.d_model),
        tgt_embed=nn.Linear(config.num_states, config.d_model),
        generator=Generator(config.d_model, 2),
        config=config,
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


class RydbergTransformer(EncoderDecoder):
    """
    RydbergTransformer is a specific implementation of the Encoder-Decoder architecture
    that uses an encoder and decoder composed of multiple layers of EncoderLayer and DecoderLayer
    modules, respectively. The encoder and decoder are followed by an embedding layer and a generator
    layer.

    Parameters:
        encoder (Encoder[EncoderLayer]): The encoder module.
        decoder (Decoder[DecoderLayer]): The decoder module.
        src_embed (nn.Module): The source embeddings module.
        tgt_embed (nn.Module): The target embeddings module.
        generator (Generator): The generator module.
        config (dict, optional): A dictionary of configuration options. Defaults to None.
        **kwargs: Additional keyword arguments.

    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: nn.Module,
        tgt_embed: nn.Module,
        generator: Generator,
        config=None,
        **kwargs,
    ):
        super().__init__(encoder, decoder, src_embed, tgt_embed, generator)
        self.config = config

    def get_log_probs(self, x, cond):
        """
        Compute the log probabilities of a given input tensor.

        Parameters:
            x (torch.Tensor): The input tensor.
            cond (torch.Tensor): The conditional tensor.

        Returns:
            torch.Tensor: The log probabilities.
        """
        log_cond_probs = self.forward(x, cond)
        log_probs = torch.einsum("bnd,bnd->b", log_cond_probs, x).sum(-1)
        return log_probs

    @property
    def psi(self):
        """
        The wave function of the model.

        Returns:
            NotImplementedError
        """
        raise NotImplementedError()

    def sample(self, x: torch.Tensor):
        """
        Generate a sample from the model.

        Parameters:
            x (torch.Tensor): The input tensor.

        Returns:
            NotImplementedError
        """
        raise NotImplementedError()

    def varloss(self, H: torch.Tensor):
        """
        Compute the variational loss.

        Parameters:
            H (torch.Tensor): The energy tensor.

        Returns:
            NotImplementedError
        """
        raise NotImplementedError()
