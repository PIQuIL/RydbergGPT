import sys

import copy
from typing import Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch_geometric.nn import GATConv, GCNConv

from rydberggpt.models.graph_embedding.models import GraphEmbedding
from rydberggpt.models.transformer.layers import DecoderLayer, EncoderLayer
from rydberggpt.models.transformer.models import (
    Decoder,
    Encoder,
    EncoderDecoder,
    Generator,
)
from rydberggpt.models.transformer.modules import (
    PositionalEncoding,
    PositionwiseFeedForward,
)
from rydberggpt.utils import to_one_hot


def get_rydberg_graph_encoder_decoder(config):
    c = copy.deepcopy
    attn = nn.MultiheadAttention(config.d_model, config.num_heads, batch_first=True)
    position = PositionalEncoding(config.d_model, config.dropout)
    ff = PositionwiseFeedForward(config.d_model, config.d_ff, config.dropout)

    model = RydbergEncoderDecoder(
        encoder=Encoder(
            EncoderLayer(config.d_model, c(attn), c(ff), config.dropout),
            config.num_blocks_encoder,
        ),
        decoder=Decoder(
            DecoderLayer(config.d_model, c(attn), c(attn), c(ff), config.dropout),
            config.num_blocks_decoder,
        ),
        src_embed=GraphEmbedding(
            graph_layer=GCNConv,  # GATConv
            in_node_dim=config.in_node_dim,
            d_hidden=config.graph_hidden_dim,
            d_model=config.d_model,
            num_layers=config.graph_num_layers,
            dropout=config.dropout,
        ),
        tgt_embed=nn.Sequential(
            nn.Linear(config.num_states, config.d_model), c(position)
        ),
        generator=Generator(config.d_model, 2),
        config=config,
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


class RydbergEncoderDecoder(EncoderDecoder):

    """
    RydbergTransformer is a specific implementation of the Encoder-Decoder architecture
    that uses an encoder and decoder composed of multiple layers of EncoderLayer and DecoderLayer
    modules, respectively. The encoder and decoder are followed by an embedding layer and a generator
    layer.

    Args:
        encoder (Encoder[EncoderLayer]): The encoder module.
        decoder (Decoder[DecoderLayer]): The decoder module.
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
    ):
        super().__init__(encoder, decoder, src_embed, tgt_embed, generator)
        self.config = config

    @torch.no_grad()
    def get_log_probs(self, x, cond):
        """
        Compute the log probabilities of a given input tensor.

        Parameters:
            x (torch.Tensor): The input tensor.
            # TODO this is not a tensor but a graph
            cond (torch.Tensor): The conditional tensor.

        Returns:
            torch.Tensor: The log probabilities.
        """
        assert x.shape[-1] == 2, "The input must be one hot encoded"

        m = torch.zeros((x.shape[0], 1, x.shape[-1]))
        m = m.to(x)
        m = torch.cat([m, x[:, :-1, :]], axis=-2)

        out = self.forward(m, cond)
        cond_log_probs = self.generator(out)

        log_probs = torch.sum(torch.sum(cond_log_probs * x, axis=-1), axis=-1)

        return log_probs

    @torch.no_grad()
    def get_samples(self, batch_size, cond, num_atoms, fmt_onehot=True):
        """
        Generate samples in one-hot encoding using the forward pass
        and sampling from the conditional probabilities.

        Args:
            batch_size (int): The number of samples to generate.
            cond (torch.Tensor): A tensor containing the input condition.
            num_atoms (int): The number of atoms to sample.
            device (str, optional): The device on which to allocate the tensors. Defaults to "cpu".

        Returns:
            torch.Tensor: A tensor containing the generated samples in one-hot encoding.
        """
        m = torch.zeros(batch_size, 1, 2, device=self.device)

        for i in range(num_atoms):
            print("{:<80}".format(f"\rGenerating atom {i+1}/{num_atoms}"), end="")
            sys.stdout.flush()

            out = self.forward(m, cond)

            cond_log_probs = self.generator(out)

            next_cond_log_probs = cond_log_probs[:, -1, :]

            next_outcome = torch.distributions.Categorical(
                logits=next_cond_log_probs
            ).sample(
                [
                    1,
                ]
            )
            next_outcome = next_outcome.reshape(next_outcome.shape[1], 1)
            next_outcome = to_one_hot(next_outcome, 2)

            m = torch.cat((m, next_outcome), dim=-2)

        for i in range(m.shape[0]):
            m[i, cond[i].num_nodes + 1 :, :] = 0

        if fmt_onehot:
            return m[:, 1:, :]
        else:
            return m[:, 1:, -1]
