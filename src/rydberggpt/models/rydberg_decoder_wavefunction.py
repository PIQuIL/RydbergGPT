import sys

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from torch_geometric.data import Batch

########################################################################################

from rydberggpt.models.transformer.models import (
    Decoder,
    Encoder,
    EncoderDecoder,
    Generator,
)
from rydberggpt.models.rydberg_encoder_decoder import RydbergEncoderDecoder
from rydberggpt.utils import to_one_hot

########################################################################################


class RydbergDecoderWavefunction(RydbergEncoderDecoder):
    def __init__(
        self,
        cond: Batch,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: nn.Module,
        tgt_embed: nn.Module,
        generator: Generator,
        config=None,
    ):
        super().__init__(
            encoder.eval(),
            decoder,
            src_embed.eval(),
            tgt_embed,
            generator,
            config,
        )

        if hasattr(cond, "num_graphs") and cond.num_graphs > 1:
            raise ValueError("cond should represent a single Hamiltonian/graph")

        self.N = cond.num_nodes
        self.cond = cond

        for p in self.encoder.parameters():
            p.requires_grad_(False)
        for p in self.src_embed.parameters():
            p.requires_grad_(False)

        memory, batch_mask = self.encode(cond)
        self.register_buffer("memory", memory)
        self.register_buffer("batch_mask", batch_mask)
        pass

    def forward(self, tgt: Tensor) -> Tensor:
        memory = self.memory.repeat([*tgt.shape[:-2], 1, 1])
        batch_mask = self.batch_mask.repeat([*tgt.shape[:-2], 1])

        return self.decode(tgt, memory, batch_mask)

    @classmethod
    def from_rydberg_encoder_decoder(
        cls, cond:Batch, model: RydbergEncoderDecoder
    ) -> RydbergDecoderWavefunction:
        """
        Create RydbergDecoderWavefunction from a RydbergEncodeDecoder model and a Hamiltonian/graph.

        Args:
            cond (Batch): The Hamiltonian/graph.
            model (RydbergEncoderDecoder): The model used to generate a RydbergDecoderWavefunction.

        Returns:
            RydbergDecoderWavefunction: The wavefunction taken from a trained RydergEncoderDecoder model for the groundstate of the Hamiltonian/graph specified by cond.

        """
        return cls(
            cond,
            model.encoder,
            model.decoder,
            model.src_embed,
            model.tgt_embed,
            model.generator,
            model.config,
        )

    pass

    def get_log_probs(self, x):
        """
        Compute the log probabilities of a given input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The log probabilities.
        """

        assert (
            len(x.shape) == 3 and x.shape[-1] == 2
        ), "The input must be one hot encoded"

        y = torch.zeros((x.shape[0], 1, x.shape[-1]))  # Initial token
        y = y.to(x)  # Match dtype and device
        y = torch.cat([y, x[:, :-1, :]], axis=-2)  # Append initial token to x

        y = self.forward(y)  # EncoderDecoder forward pass
        y = self.generator(y)  # Conditional log probs

        y = torch.sum(torch.sum(y * x, axis=-1), axis=-1)  # Log prob of full x

        return y

    def get_samples(
        self, batch_size, fmt_onehot=True, requires_grad=False, verbose=True
    ):
        """
        Generate samples using the forward pass and sampling from the conditional probabilities.
        The samples can be returned either in one-hot encoding format or in label format,
        according to the `fmt_onehot` argument.

        Args:
            batch_size (int): The number of samples to generate.
            fmt_onehot (bool, optional): A flag to indicate whether to return the samples
              in one-hot encoding format. If False, the samples are returned in label format. Defaults to True.
            requires_grad (bool, optional): A flag to determine if grad is needed when sampling. Defaults to False,
            verbose (bool, optional): A flag indicating whether to print sampling progress. Defaults to True,

        Returns:
            torch.Tensor: A tensor containing the generated samples. The shape of the tensor is (batch_size, num_atoms, 2) for one-hot encoding format, and (batch_size, num_atoms) for label format. The samples are padded according to the number of nodes in each graph within `cond`.
        """
        if verbose:
            print("")

        num_atoms = self.N

        m = torch.zeros(batch_size, 1, 2, device=self.device)

        for i in range(num_atoms):
            if verbose:
                print("{:<80}".format(f"\rGenerating atom {i+1}/{num_atoms}"), end="")
                sys.stdout.flush()

            y = self.forward(m)  # EncoderDecoder forward pass
            y = self.generator(y)  # Conditional log probs
            y = y[:, -1, :]  # Next conditional log probs

            if requires_grad:
                y = F.gumbel_softmax(logits=y, tau=1, hard=True)[..., None, :]

            else:
                y = torch.distributions.Categorical(logits=y).sample(
                    [
                        1,
                    ]
                )  # Sample from next conditional log probs
                y = y.reshape(y.shape[1], 1)  # Reshape
                y = to_one_hot(y, 2)  # Convert from label to one hot encoding

            m = torch.cat((m, y), dim=-2)  # Append next sample to tensor

        if fmt_onehot:
            m = m[:, 1:, :]  # Remove initial token
        else:
            m = m[:, 1:, -1]  # Remove initial token and put into label format

        if verbose:
            print("")
        return m

    def get_x_magnetization(
        self,
        samples: torch.Tensor,  # dtype=torch.int64
    ):
        """
        Calculates x magnetization of the model.

        Args:
            samples (torch.Tensor): Samples drawn from model based on cond.

        Returns:
            torch.Tensor: A tensor containing the estimated x magnetization of each sample.
        """

        # Create all possible states achievable by a single spin flip
        flipped = (samples[:, None, :] + torch.eye(samples.shape[-1])[None, ...]) % 2
        flipped = flipped.reshape(-1, samples.shape[-1])

        # Get propabilities of sampled states and the single spin flipped states
        sample_log_probs = self.get_log_probs(to_one_hot(samples, 2))
        flipped_log_probs = self.get_log_probs(to_one_hot(flipped, 2))
        flipped_log_probs = flipped_log_probs.reshape(-1, samples.shape[-1])

        # Calculate ratio of the wavefunction for the sampled and flipped states
        log_psi_ratio = 0.5 * (flipped_log_probs - sample_log_probs[:, None])
        psi_ratio = torch.exp(log_psi_ratio)

        x_magnetization = psi_ratio.sum(-1)
        return x_magnetization

    def get_rydberg_energy(
        self,
        samples: torch.Tensor,  # dtype=torch.int64
        undo_sample_path=None,
        undo_sample_path_args=None,
    ) -> torch.Tensor:
        """
        Calculates energy of the model based on the Hamiltonian defined by cond (graph).

        Args:
            samples (torch.Tensor): Samples drawn from model based on cond.
           undo_sample_path (torch.Tensor): Map that undoes the sample path of the model to match the labelling of in the graph.
           undo_sample_path_args (tuple): Additional arguments for undo_sample_path.

        Returns:
            torch.Tensor: A tensor containing the estimated energy of each sample alongside its decomposition into terms.
        """

        model = self
        samples = samples
        cond = self.cond

        delta = cond.x[:, 0]  # Detuning coeffs
        omega = cond.x[0, 1]  # Rabi frequency
        beta = cond.x[0, 2]
        Rb = cond.x[0, 3]  # blockade radius

        ########################################################################################

        # Estimate interaction/Rydberg blockade term

        if undo_sample_path is not None:
            unpathed_samples = undo_sample_path(samples, *undo_sample_path_args)
        else:
            unpathed_samples = samples

        interaction = (
            (
                unpathed_samples[..., cond.edge_index].prod(dim=-2)
                * cond.edge_attr[None, ...]
            ).sum(dim=-1)
            * Rb**6
            * omega
        )

        # Estimate detuning term
        detuning = (delta * unpathed_samples).sum(1)  # sum over sequence length

        # Estimate sigma_x
        x_magnetization = self.get_x_magnetization(samples)
        offdiag_energy = 0.5 * omega * x_magnetization

        # Diagonal part of energy
        diag_energy = interaction - detuning

        energy = diag_energy + offdiag_energy  # Energy estimate

        return torch.stack(
            [
                energy,
                interaction,
                detuning,
                offdiag_energy,
            ]
        ).T

    def variational_loss(self, batch_size, undo_sample_path, undo_sample_path_args):
        samples = self.get_samples(
            batch_size=batch_size, fmt_onehot=False, requires_grad=True, verbose=False
        )

        N = self.N
        omega = self.cond.x[0, 1]

        energy = self.get_rydberg_energy(
            samples=samples,
            undo_sample_path=undo_sample_path,
            undo_sample_path_args=undo_sample_path_args,
        )[..., 0].mean() / (N * omega)

        return energy
