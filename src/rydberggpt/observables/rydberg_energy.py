from typing import Callable

import torch

from rydberggpt.models.rydberg_encoder_decoder import RydbergEncoderDecoder
from rydberggpt.utils import to_one_hot


# TODO test function
@torch.no_grad()
def get_rydberg_energy(
    model: RydbergEncoderDecoder,
    samples: torch.Tensor,  # dtype=torch.int64
    cond: torch.Tensor,  # dtype=torch.float32
    device: torch.device,
    undo_sample_path: None,
) -> torch.Tensor:
    """
    Calculates energy of the model based on the Hamiltonian defined by cond (graph).

    Args:
        model (RydbergEncoderDecoder): Model to estimate energy on
        samples (torch.Tensor): Samples drawn from model based on cond.
        cond (torch.Tensor): A tensor containing the input condition.
        device (str, optional): The device on which to allocate the tensors. Defaults to "cpu".
       undo_sample_path (torch.Tensor): Map that undoes the sample path of the model to match the labelling of in the graph.

    Returns:
        torch.Tensor: A tensor containing the generated samples in one-hot encoding.
    """

    model = model.to(device)
    samples = samples.to(device)
    cond = cond.to(device)

    delta = cond.x[:, 0]  # Detuning coeffs
    omega = cond.x[0, 1]  # Rabi frequency
    beta = cond.x[0, 2]
    Rb = cond.x[0, 3]  # blockade radius

    if undo_sample_path is not None:
        samples = undo_sample_path(samples)

    ########################################################################################

    # Interaction/Rydberg blockade term
    interaction = (
        (samples[..., cond.edge_index].prod(dim=-2) * cond.edge_attr[None, ...]).sum(
            dim=-1
        )
        * Rb**6
        * omega
    )

    detuning = (delta * samples).sum(1)  # sum over sequence length

    # Estimate sigma_x
    flipped = (samples[:, None, :] + torch.eye(samples.shape[-1])[None, ...]) % 2
    flipped = flipped.reshape(-1, samples.shape[-1])

    sample_log_probs = model.get_log_probs(to_one_hot(samples, 2), cond)
    flipped_log_probs = model.get_log_probs(to_one_hot(flipped, 2), cond)
    flipped_log_probs = flipped_log_probs.reshape(-1, samples.shape[-1])

    log_psi_ratio = 0.5 * (flipped_log_probs - sample_log_probs[:, None])

    psi_ratio = torch.exp(log_psi_ratio)

    offdiag_energy = -(omega * 0.5 * psi_ratio).sum(-1)

    # Diagonal part of energy
    diag_energy = interaction - detuning

    energy = diag_energy + offdiag_energy  # Energy estimate

    return torch.stack([energy, interaction, detuning, offdiag_energy]).T
