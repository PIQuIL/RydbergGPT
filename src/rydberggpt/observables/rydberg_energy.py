from typing import Callable

import torch

from rydberggpt.models.rydberg_encoder_decoder import RydbergEncoderDecoder
from rydberggpt.utils import to_one_hot


# TODO refactor this function
# TODO test function
# TODO use schuylers fast code for energy calculation
# TODO do we need to worry about the sampling path? (i.e. snake path)
@torch.no_grad()
def get_rydberg_energy(
    model: RydbergEncoderDecoder,
    V: torch.Tensor,  # dtype=torch.float64
    omega: torch.Tensor,  # dtype=torch.float32
    delta: torch.Tensor,  # dtype=torch.float32
    samples: torch.Tensor,  # dtype=torch.int64
    cond: torch.Tensor,  # dtype=torch.float32
    # log_probs: torch.Tensor,  # dtype=torch.float32
    num_atoms: int,
    device: torch.device,
) -> torch.Tensor:

    assert (
        V.shape[1] == samples.shape[1]
    ), "Interaction matrix and number of atoms do not match."
    # remove batch dim of V
    V = V.to(device)

    samples = samples.to(V)  # Convert samples tensor to float64

    interaction = torch.einsum("ij,bi,bj->b", V, samples, samples)  # [batch_size]
    detuning = (delta * samples).sum(1)  # sum over sequence length

    # Estimate sigma_x
    flipped = (samples[:, None, :] + torch.eye(samples.shape[-1])[None, ...]) % 2
    flipped = flipped.reshape(-1, samples.shape[-1])

    flipped_onehot = to_one_hot(flipped, 2)
    samples_onehot = to_one_hot(samples, 2)

    sample_log_probs = model.get_log_probs(samples_onehot, cond)
    flipped_log_probs = model.get_log_probs(flipped_onehot, cond)

    flipped_log_probs = flipped_log_probs.reshape(-1, samples.shape[-1])
    log_probs_ratio = flipped_log_probs - sample_log_probs[:, None]
    probs_ratio = torch.exp(log_probs_ratio)
    sigma_x = probs_ratio

    diag_energy = interaction - detuning
    offdiag_energy = -(omega / 2 * sigma_x).sum(-1)

    energy = diag_energy + offdiag_energy

    return energy
