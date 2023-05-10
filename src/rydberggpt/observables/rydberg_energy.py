from typing import Callable

import torch

from rydberggpt.utils import to_one_hot


# TODO refactor this function
# TODO test function
# TODO use schuylers fast code for energy calculation
# TODO do we need to worry about the sampling path? (i.e. snake path)
@torch.no_grad()
def get_rydberg_energy(
    get_log_probs: Callable,
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
    detuning = delta * (samples.sum(1))  # sum over sequence length

    samples = samples.to(torch.int64)
    # sigma x
    offdiagonal_lp = torch.zeros(
        samples.shape[0], samples.shape[1], dtype=torch.double
    )  # [batch_size, seq_len]

    # get sigma x
    for i in range(num_atoms):
        samples[:, i].bitwise_xor_(True)  # [batch_size, seq_len]
        log_probs = get_log_probs(to_one_hot(samples, 2), cond)  # [batch_size]
        offdiagonal_lp[:, i] = log_probs  # [batch_size, seq_len]
        samples[:, i].bitwise_xor_(True)  # [batch_size, seq_len]

    temp = torch.logsumexp(offdiagonal_lp / 2, dim=1)  # [batch_size]
    offdiagonal_lp = temp - log_probs / 2
    # end

    energy = interaction - (omega / 2) * (offdiagonal_lp.t().exp()) - detuning
    return energy

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
