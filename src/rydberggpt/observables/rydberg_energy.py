from typing import Callable

import torch

from rydberggpt.models.rydberg_encoder_decoder import RydbergEncoderDecoder
from rydberggpt.utils import to_one_hot


# TODO test function
@torch.no_grad()
def get_rydberg_energy(
    model: RydbergEncoderDecoder,
    V: torch.Tensor,  # dtype=torch.float64
    samples: torch.Tensor,  # dtype=torch.int64
    cond: torch.Tensor,  # dtype=torch.float32
    # log_probs: torch.Tensor,  # dtype=torch.float32
    device: torch.device,
) -> torch.Tensor:
    """
    Calculates energy of the model based on the Hamiltonian defined by cond (graph).

    Args:
        model (RydbergEncoderDecoder): Model to estimate energy on
        V (torch.Tensor): Interaction matrix, ensure the interaction matrix is defined in terms of the sampling path permutation.
        samples (torch.Tensor): Samples drawn from model based on cond.
        cond (torch.Tensor): A tensor containing the input condition.
        device (str, optional): The device on which to allocate the tensors. Defaults to "cpu".

    Returns:
        torch.Tensor: A tensor containing the generated samples in one-hot encoding.
    """

    assert (
        V.shape[1] == samples.shape[1]
    ), "Interaction matrix and number of atoms do not match."
    # remove batch dim of V
    V = V.to(device)

    samples = samples.to(V)  # Match dtype and device
    cond = cond.to(V.device)  # Match dtype and device
    delta = cond.x[:, 0]  # Detuning coeff
    omega = cond.x[:, 1]  # Rabi frequency

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

    interaction = 0.5 * torch.einsum("ij,bi,bj->b", V, samples, samples)  # [batch_size]
    detuning = (delta * samples).sum(1)  # sum over sequence length

    diag_energy = interaction - detuning

    energy = diag_energy + offdiag_energy  # Energy estimate

    return energy
