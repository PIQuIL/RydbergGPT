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
            cond (torch.Tensor): The conditional tensor.

        Returns:
            torch.Tensor: The log probabilities.
        """
        assert (
            x.shape[-1] == 2
        ), "This function requires the input to be one hot encoded"
        out = self.forward(x, cond)
        cond_log_probs = self.generator(out)
        log_probs = torch.sum(torch.sum(cond_log_probs * x, axis=-1), axis=-1)
        return log_probs

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
            out = self.forward(m_onehot, cond)  # [batch_size, i, d_model]
            cond_log_probs = self.generator(out)  # [batch_size, i, 2]
            cond_probs = torch.exp(cond_log_probs[:, i, :])  # [batch_size, 2]
            # full_cond_probs = torch.exp(cond_log_probs)  # [batch_size, 2] for debugging
            next_outcome = torch.multinomial(
                cond_probs, num_samples=1
            )  # [batch_size, 1]
            next_outcome_onehot = to_one_hot(next_outcome, 2)
            m_onehot = torch.cat((m_onehot, next_outcome_onehot), dim=1)
            cond_log_probs[:, i] = torch.gather(
                cond_log_probs[:, i, :], 1, next_outcome
            ).squeeze(1)
        # remove the first column of zeros (the initial state)
        return m_onehot[:, 1:, :], cond_log_probs

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
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
                - The first tensor contains the generated samples.
                - The second tensor contains the log probabilities of the samples.
        """
        samples, _ = self.get_samples_and_log_probs(batch_size, cond, num_atoms)
        return samples

    @torch.no_grad()
    def get_samples_and_log_probs(
        self, batch_size: int, cond: Tensor, num_atoms: int
    ) -> Tuple[Tensor, Tensor]:
        """
        Generate samples and their log probabilities using the forward pass and sampling
        from the conditional probabilities. Sampling takes place using the snake convention.

        Args:
            batch_size (int): The number of samples to generate.
            cond (torch.Tensor): A tensor containing the input condition.
            num_atoms (int): The number of atoms to sample or sequence length.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
                - The first tensor contains the generated samples.
                - The second tensor contains the log probabilities of the samples.
        """
        samples_one_hot, cond_log_probs = self.get_samples_one_hot(
            batch_size, cond, num_atoms
        )
        samples = torch.argmax(samples_one_hot, dim=-1)
        log_probs = torch.sum(cond_log_probs, dim=-1)  # [batch_size, seq_len]
        return samples, log_probs

    # @torch.no_grad()
    # def get_energy(
    #     self,
    #     V: torch.Tensor,  # dtype=torch.float64
    #     omega: torch.Tensor,  # dtype=torch.float32
    #     delta: torch.Tensor,  # dtype=torch.float32
    #     samples: torch.Tensor,  # dtype=torch.int64
    #     cond: torch.Tensor,  # dtype=torch.float32
    #     log_probs: torch.Tensor,  # dtype=torch.float32
    #     num_atoms: int,
    # ) -> torch.Tensor:
    #     assert (
    #         V.shape[1] == samples.shape[1]
    #     ), "Interaction matrix and number of atoms do not match."
    #     # remove batch dim of V
    #     V = V.squeeze(0).to(self.device)
    #     samples = samples.to(V)  # Convert samples tensor to float64

    #     interaction = torch.einsum("ij,bi,bj->b", V, samples, samples)  # [batch_size]
    #     detuning = delta * (samples.sum(1))  # sum over sequence length
    #     offdiagonal_lp = torch.zeros(
    #         samples.shape[0], samples.shape[1], dtype=torch.double
    #     )  # [batch_size, seq_len]

    #     for i in range(num_atoms):
    #         samples[:, i].bitwise_xor_(True)  # [batch_size, seq_len]
    #         log_probs = self.get_log_probs(to_one_hot(samples, 2), cond)  # [batch_size]
    #         offdiagonal_lp[:, i] = log_probs  # [batch_size, seq_len]
    #         samples[:, i].bitwise_xor_(True)  # [batch_size, seq_len]

    #     temp = torch.logsumexp(offdiagonal_lp / 2, dim=1)  # [batch_size]
    #     offdiagonal_lp = temp - log_probs / 2

    #     energy = interaction - (omega / 2) * (offdiagonal_lp.t().exp()) - detuning
    #     return energy

    # def Rydberg_Energy_Vectorized(
    #     self,
    #     O,
    #     d,
    #     V0,
    #     O_mat,  # what is this?
    #     V_mat,  # what is this?
    #     coeffs,
    #     samples,  # [batch_size, seq_len]
    #     sample_log_probs,  # [batch_size, seq_len, 2]
    # ):
    #     # samples are in binary
    #     # samples [batch_size, seq_len]
    #     numsamples = samples.shape[0]
    #     N = samples.shape[1]
    #     samples = samples.float()
    #     energies_delta = torch.sum(samples, axis=1)

    #     ni_plus_nj = V_mat @ samples.t()
    #     ni_nj = torch.floor(ni_plus_nj / 2)
    #     c_ni_nj = coeffs * ni_nj
    #     energies_V = torch.sum(c_ni_nj, axis=0)

    #     samples_tiled_not_flipped = samples.unsqueeze(2).repeat(1, 1, N)
    #     samples_tiled_flipped = (samples_tiled_not_flipped + O_mat.t()[None, :, :]) % 2
    #     samples_tiled_flipped = samples_tiled_flipped.transpose(1, 2)
    #     samples_tiled_flipped = samples_tiled_flipped.reshape(numsamples * N, N)
    #     samples_tiled_flipped = samples_tiled_flipped.long()

    #     log_probs_flipped = self.get_log_probs(samples_tiled_flipped)
    #     log_probs_flipped = log_probs_flipped.reshape(numsamples, N)
    #     log_prob_ratio = torch.exp(log_probs_flipped - sample_log_probs[:, None])
    #     energies_O = torch.sum(log_prob_ratio, axis=1)

    #     Energies = -1 * d * energies_delta + V0 * energies_V - (1 / 2) * O * energies_O
    #     return Energies
