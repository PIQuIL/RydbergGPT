import copy
import sys

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv

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
    def get_log_probs(self, x: torch.Tensor, cond: Batch):
        """
        Compute the log probabilities of a given input tensor.

        Parameters:
            x (torch.Tensor): The input tensor.
            cond (Batch): The conditional graph structure.

        Returns:
            (torch.Tensor): The log probabilities.
        """

        if not hasattr(cond, "num_graphs"):
            cond = Batch.from_data_list([cond.clone() for _ in range(len(x))])

        assert (
            len(x.shape) == 3 and x.shape[-1] == 2
        ), "The input must be one hot encoded"

        y = torch.zeros((x.shape[0], 1, x.shape[-1]))  # Initial token
        y = y.to(x)  # Match dtype and device
        y = torch.cat([y, x[:, :-1, :]], axis=-2)  # Append initial token to x

        y = self.forward(y, cond)  # EncoderDecoder forward pass
        y = self.generator(y)  # Conditional log probs

        y = torch.sum(torch.sum(y * x, axis=-1), axis=-1)  # Log prob of full x

        return y

    @torch.no_grad()
    def get_samples(
        self,
        batch_size: int,
        cond: Batch,
        num_atoms: int,
        fmt_onehot: bool = True,
    ):
        """
        Generate samples using the forward pass and sampling from the conditional probabilities.
        The samples can be returned either in one-hot encoding format or in label format,
        according to the `fmt_onehot` argument.

        Args:
            batch_size (int): The number of samples to generate.
            cond (torch_geometric.data.Batch): The batch of conditional graph structures.
            num_atoms (int): The number of atoms to sample. For num_atoms > num_nodes
              in each graph within `cond`, the extra atoms are padded with zeros (onehot) or nan (label).
            fmt_onehot (bool, optional): A flag to indicate whether to return the samples
              in one-hot encoding format. If False, the samples are returned in label format. Defaults to True.

        Returns:
            (torch.Tensor): A tensor containing the generated samples. The shape of the tensor is (batch_size, num_atoms, 2) for one-hot encoding format, and (batch_size, num_atoms) for label format. The samples are padded according to the number of nodes in each graph within `cond`.
        """

        if not hasattr(cond, "num_graphs"):
            cond = Batch.from_data_list([cond.clone() for _ in range(batch_size)])

        assert (
            cond.num_graphs == batch_size
        ), "Incompatible arguments, batch_size ({}) does not match cond.num_graphs ({})".format(
            batch_size, cond.num_graphs
        )

        m = torch.zeros(batch_size, 1, 2, device=self.device)

        for i in range(num_atoms):
            print("{:<80}".format(f"\rGenerating atom {i+1}/{num_atoms}"), end="")
            sys.stdout.flush()

            y = self.forward(m, cond)  # EncoderDecoder forward pass
            y = self.generator(y)  # Conditional log probs
            y = y[:, -1, :]  # Next conditional log probs
            y = torch.distributions.Categorical(logits=y).sample(
                [
                    1,
                ]
            )  # Sample from next conditional log probs
            y = y.reshape(y.shape[1], 1)  # Reshape
            y = to_one_hot(y, 2)  # Convert from label to one hot encoding

            m = torch.cat((m, y), dim=-2)  # Append next sample to tensor

        if fmt_onehot:
            for i in range(m.shape[0]):
                # Depending on num_nodes/num_atoms in graph pad samples with [0,0]
                m[i, cond[i].num_nodes + 1 :, :] = 0

            m = m[:, 1:, :]  # Remove initial token
        else:
            m = m[:, :, -1]

            for i in range(m.shape[0]):
                # Depending on num_nodes/num_atoms in graph pad samples with nan
                m[i, cond[i].num_nodes + 1 :] = torch.nan

            m = m[:, 1:]

        print("")
        return m
