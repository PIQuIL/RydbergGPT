import sys

import torch
from torch import Tensor, nn

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
        cond,
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

        memory, batch_mask = self.encode(cond)
        self.register_buffer("memory", memory)
        self.register_buffer("batch_mask", batch_mask)
        pass

    def forward(self, tgt: Tensor) -> Tensor:
        memory = self.memory.repeat([*tgt.shape[:-2], 1, 1])
        batch_mask = self.batch_mask.repeat([*tgt.shape[:-2], 1])

        return self.decode(tgt, memory, batch_mask)

    @classmethod
    def from_rydberg_encoder_decoder(cls, cond, model: RydbergEncoderDecoder):
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

    @torch.no_grad()
    def get_log_probs(self, x):
        """
        Compute the log probabilities of a given input tensor.

        Parameters:
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

    @torch.no_grad()
    def get_samples(self, batch_size, fmt_onehot=True):
        """
        Generate samples using the forward pass and sampling from the conditional probabilities.
        The samples can be returned either in one-hot encoding format or in label format,
        according to the `fmt_onehot` argument.

        Args:
            batch_size (int): The number of samples to generate.
            fmt_onehot (bool, optional): A flag to indicate whether to return the samples
              in one-hot encoding format. If False, the samples are returned in label format.
              Defaults to True.

        Returns:
            torch.Tensor: A tensor containing the generated samples. The shape of the tensor is (batch_size, num_atoms, 2) for one-hot encoding format, and (batch_size, num_atoms) for label format. The samples are padded according to the number of nodes in each graph within `cond`.
        """

        num_atoms = self.N

        m = torch.zeros(batch_size, 1, 2, device=self.device)

        for i in range(num_atoms):
            print("{:<80}".format(f"\rGenerating atom {i+1}/{num_atoms}"), end="")
            sys.stdout.flush()

            y = self.forward(m)  # EncoderDecoder forward pass
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
            m = m[:, 1:, :]  # Remove initial token
        else:
            m = m[:, 1:, -1]  # Remove initial token and put into label format

        print("")
        return m

    @torch.no_grad()
    def get_x_magnetization(
        self,
        samples: torch.Tensor,  # dtype=torch.int64
        device: torch.device,
    ):
        """
        Calculates x magnetization of the model.

        Args:
            samples (torch.Tensor): Samples drawn from model based on cond.
            device (str, optional): The device on which to allocate the tensors. Defaults to "cpu".

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

    @torch.no_grad()
    def get_rydberg_energy(
        self,
        samples: torch.Tensor,  # dtype=torch.int64
        device: torch.device,
        undo_sample_path=None,
        undo_sample_path_args=None,
    ) -> torch.Tensor:
        """
        Calculates energy of the model based on the Hamiltonian defined by cond (graph).

        Args:
            samples (torch.Tensor): Samples drawn from model based on cond.
            device (str, optional): The device on which to allocate the tensors. Defaults to "cpu".
           undo_sample_path (torch.Tensor): Map that undoes the sample path of the model to match the labelling of in the graph.
           undo_sample_path_args (tuple): Additional arguments for undo_sample_path.

        Returns:
            torch.Tensor: A tensor containing the estimated energy of each sample alongside its decomposition into terms.
        """

        model = self.to(device)
        samples = samples.to(device)
        cond = self.cond.to(device)

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
        x_magnetization = self.get_x_magnetization(samples, device)
        offdiag_energy = -0.5 * omega * x_magnetization

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
