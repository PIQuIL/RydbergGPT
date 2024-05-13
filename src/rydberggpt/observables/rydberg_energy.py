import numpy as np
import torch

from rydberggpt.models.rydberg_encoder_decoder import RydbergEncoderDecoder
from rydberggpt.utils import to_one_hot


@torch.no_grad()
def get_staggered_magnetization(
    samples: torch.Tensor,
    Lx: int,
    Ly: int,
    device: torch.device,
    undo_sample_path=None,
    undo_sample_path_args=None,
):
    """
    Calculates staggered magnetization of the model.

    Args:
        samples (torch.Tensor): Samples drawn from model.
        Lx (int): Linear size in the x dimension
        Ly (int): Linear size in the y dimension
        device (str, optional): The device on which to allocate the tensors. Defaults to "cpu".
        undo_sample_path (torch.Tensor): Map that undoes the sample path of the model to match the labelling of in the graph.
        undo_sample_path_args (tuple): Additional arguments for undo_sample_path.

    Returns:
        (torch.Tensor): A tensor containing the estimated staggered magnetization of each sample.
    """

    if undo_sample_path is not None:
        unpathed_samples = undo_sample_path(samples, *undo_sample_path_args)
    else:
        unpathed_samples = samples

    unpathed_samples = unpathed_samples.reshape(-1, Ly, Lx)

    unpathed_sigmas = 2 * unpathed_samples - 1

    idcs = np.indices((Ly, Lx))
    checkerboard = 2 * (idcs.sum(0) % 2) - 1
    checkerboard = torch.from_numpy(checkerboard).to(device=device)

    staggered_magnetization = torch.abs((checkerboard * unpathed_sigmas).mean((-1, -2)))

    return staggered_magnetization


@torch.no_grad()
def get_x_magnetization(
    model: RydbergEncoderDecoder,
    samples: torch.Tensor,  # dtype=torch.int64
    cond: torch.Tensor,  # dtype=torch.float32
    device: torch.device,
):
    """
    Calculates x magnetization of the model.

    Args:
        model (RydbergEncoderDecoder): Model to estimate energy on.
        samples (torch.Tensor): Samples drawn from model based on cond.
        cond (torch.Tensor): A tensor containing the input condition.
        device (str, optional): The device on which to allocate the tensors. Defaults to "cpu".

    Returns:
        (torch.Tensor): A tensor containing the estimated x magnetization of each sample.
    """

    model = model.to(device)
    samples = samples.to(device)
    cond = cond.to(device)

    # Create all possible states achievable by a single spin flip
    flipped = (samples[:, None, :] + torch.eye(samples.shape[-1])[None, ...]) % 2
    flipped = flipped.reshape(-1, samples.shape[-1])

    # Get propabilities of sampled states and the single spin flipped states
    sample_log_probs = model.get_log_probs(to_one_hot(samples, 2), cond)
    flipped_log_probs = model.get_log_probs(to_one_hot(flipped, 2), cond)
    flipped_log_probs = flipped_log_probs.reshape(-1, samples.shape[-1])

    # Calculate ratio of the wavefunction for the sampled and flipped states
    log_psi_ratio = 0.5 * (flipped_log_probs - sample_log_probs[:, None])
    psi_ratio = torch.exp(log_psi_ratio)

    x_magnetization = psi_ratio.sum(-1)
    return x_magnetization


@torch.no_grad()
def get_rydberg_energy(
    model: RydbergEncoderDecoder,
    samples: torch.Tensor,  # dtype=torch.int64
    cond: torch.Tensor,  # dtype=torch.float32
    device: torch.device,
    undo_sample_path=None,
    undo_sample_path_args=None,
) -> torch.Tensor:
    """
    Calculates energy of the model based on the Hamiltonian defined by cond (graph).

    Args:
        model (RydbergEncoderDecoder): Model to estimate energy on.
        samples (torch.Tensor): Samples drawn from model based on cond.
        cond (torch.Tensor): A tensor containing the input condition.
        device (str, optional): The device on which to allocate the tensors. Defaults to "cpu".
        undo_sample_path (torch.Tensor): Map that undoes the sample path of the model to match the labelling of in the graph.
        undo_sample_path_args (tuple): Additional arguments for undo_sample_path.

    Returns:
        (torch.Tensor): A tensor containing the estimated energy of each sample alongside its decomposition into terms.
    """

    model = model.to(device)
    samples = samples.to(device)
    cond = cond.to(device)

    delta = cond.x[:, 0]  # Detuning coeffs
    omega = cond.x[0, 1]  # Rabi frequency
    # beta = cond.x[0, 2]  # Inverse temperature
    Rb = cond.x[0, 3]  # Rydberg Blockade radius

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

    detuning = (delta * unpathed_samples).sum(1)  # sum over sequence length

    x_magnetization = get_x_magnetization(model, samples, cond, device)

    offdiag_energy = 0.5 * omega * x_magnetization
    diag_energy = interaction - detuning
    energy = diag_energy - offdiag_energy

    return torch.stack(
        [
            energy,
            interaction,
            detuning,
            diag_energy,
            offdiag_energy,
        ]
    ).T
