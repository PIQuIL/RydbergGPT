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
        out = self.forward(x, cond)
        cond_log_probs = self.generator(out)
        log_probs = torch.sum(torch.sum(cond_log_probs * x, axis=-1), axis=-1)
        return log_probs

    # TODO add support for batchsize and number_batches
    # the output should be a tensor concatenated along the batch dimension
    @torch.no_grad()
    def get_samples_one_hot(
        self, batch_size: int, cond: Tensor, num_atoms: int
    ) -> Tuple[Tensor, Tensor]:
        """
        Generate samples in one-hot encoding and their log probabilities using the forward pass
        and sampling from the conditional probabilities.

        Args:
            batch_size (int): The number of samples to generate.
            cond (torch.Tensor): A tensor containing the input condition.
            num_atoms (int): The number of atoms to sample.
            device (str, optional): The device on which to allocate the tensors. Defaults to "cpu".

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
                - The first tensor contains the generated samples in one-hot encoding.
                - The second tensor contains the log probabilities of the samples.
        """
        self.eval()
        m_onehot = torch.zeros(batch_size, 1, 2, device=self.device)
        cond_log_probs = torch.zeros(batch_size, num_atoms, device=self.device)
        for i in range(num_atoms):
            print(f"Generating atom {i+1}/{num_atoms}", end="\r")
            out = self.forward(m_onehot, cond)  # [batch_size, i, d_model]
            cond_log_probs = self.generator(out)  # [batch_size, i, 2]
            cond_probs = torch.exp(cond_log_probs[:, i, :])  # [batch_size, 2]
            next_outcome = torch.multinomial(cond_probs, 1)  # [batch_size, 1]
            next_outcome_onehot = to_one_hot(next_outcome, 2)
            m_onehot = torch.cat((m_onehot, next_outcome_onehot), dim=1)
        return m_onehot[:, 1:, :]

    @torch.no_grad()
    def get_samples_one_hot(
        self, batch_size: int, cond: Tensor, num_atoms: int
    ) -> Tensor:
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
        self.eval()
        m_onehot = torch.zeros(batch_size, 1, 2, device=self.device)

        for i in range(num_atoms):
            print(f"Generating atom {i+1}/{num_atoms}", end="\r")
            out = self.forward(m_onehot, cond)  # [batch_size, i, d_model]
            cond_log_probs = self.generator(out)  # [batch_size, i, 2]
            cond_probs = torch.exp(cond_log_probs[:, i, :])  # [batch_size, 2]
            next_outcome = torch.multinomial(cond_probs, 1)  # [batch_size, 1]
            next_outcome_onehot = to_one_hot(next_outcome, 2)
            m_onehot = torch.cat((m_onehot, next_outcome_onehot), dim=1)

        return m_onehot[:, 1:, :]

    @torch.no_grad()
    def get_samples_one_hot_batched(
        self, batch_size: int, number_batches: int, cond: Tensor, num_atoms: int
    ) -> Tensor:
        """
        Generate samples in one-hot encoding for the specified number of batches.

        Args:
            batch_size (int): The number of samples to generate per batch.
            number_batches (int): The number of batches to generate.
            cond (torch.Tensor): A tensor containing the input condition.
            num_atoms (int): The number of atoms to sample.
            device (str, optional): The device on which to allocate the tensors. Defaults to "cpu".

        Returns:
            torch.Tensor: A tensor containing the generated samples in one-hot encoding.
        """
        samples_list = []

        for batch in range(number_batches):
            print(f"Generating batch {batch + 1}/{number_batches}")
            batch_samples = self.get_samples_one_hot(batch_size, cond, num_atoms)
            samples_list.append(batch_samples)

        # Concatenate the generated samples along the batch dimension
        all_samples = torch.cat(samples_list, dim=0)

        return all_samples

    # TODO add support for batchsize and number_batches,
    # the output should be a tensor concatenated along the batch dimension
    @torch.no_grad()
    def get_samples(
        self, batch_size: int, cond: Tensor, num_atoms: int
    ) -> Tuple[Tensor, Tensor]:
        """
        Generate samples and their log probabilities using the forward pass and sampling
        from the conditional probabilities. Sampling takes place using the snake convention.

        Args:
            batch_size (int): The number of samples to generate.
            cond (torch.Tensor): A tensor containing the input condition.
            num_atoms (int): The number of atoms to sample.

        Returns:
            torch.Tensor: the generated samples.
        """
        samples_one_hot = self.get_samples_one_hot(batch_size, cond, num_atoms)
        samples = torch.argmax(samples_one_hot, dim=-1)
        return samples

    # samples, _ = self.get_samples_and_log_probs(batch_size, cond, num_atoms)
    # @torch.no_grad()
    # def get_samples_and_log_probs(
    #     self, batch_size: int, cond: Tensor, num_atoms: int
    # ) -> Tuple[Tensor, Tensor]:
    #     """
    #     Generate samples and their log probabilities using the forward pass and sampling
    #     from the conditional probabilities. Sampling takes place using the snake convention.

    #     Args:
    #         batch_size (int): The number of samples to generate.
    #         cond (torch.Tensor): A tensor containing the input condition.
    #         num_atoms (int): The number of atoms to sample or sequence length.

    #     Returns:
    #         Tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
    #             - The first tensor contains the generated samples.
    #             - The second tensor contains the log probabilities of the samples.
    #     """
    #     samples_one_hot, cond_log_probs = self.get_samples_one_hot(
    #         batch_size, cond, num_atoms
    #     )
    #     samples = torch.argmax(samples_one_hot, dim=-1)
    #     log_probs = torch.sum(cond_log_probs, dim=-1)  # [batch_size, seq_len]
    #     return samples, log_probs
