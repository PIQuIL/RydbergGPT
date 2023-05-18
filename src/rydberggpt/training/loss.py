import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# TODO if we decide to use padding we have to modify this function.
# If paddind is detected, meaning the start or stop token [0,0] is present,
# we ignore the rest for computing the loss.
class NLLLoss(pl.LightningModule):
    """
    KLLoss is a custom loss function class that computes the Kullback-Leibler divergence
    between the target distribution (tgt) and the conditional log probabilities
    (cond_log_probs).
    """

    def __init__(self):
        super(NLLLoss, self).__init__()
        # self.criterion = nn.KLDivLoss(reduction="batchmean")

    def forward(self, cond_log_probs: Tensor, tgt: Tensor) -> Tensor:
        """
        Computes the Kullback-Leibler divergence loss between the target distribution (tgt)
        and the conditional log probabilities (cond_log_probs).

        Args:
            cond_log_probs (torch.Tensor): The conditional log probabilities tensor
                                            with shape (batch_size, seq_length, vocab_size).
            tgt (torch.Tensor): The target distribution tensor with shape
                                 (batch_size, seq_length, vocab_size).

        Returns:
            torch.Tensor: The computed Kullback-Leibler divergence loss, a scalar tensor.
        """

        # TODO Given the target we look for the index where we reach a [0,0] token
        # if reached mask all the cond_log_probs after that index to zero.
        # Then compute the loss.

        num_atoms = tgt.shape[-2] - (tgt == 0.0).all(-1).sum(-1)

        log_probs = (cond_log_probs * tgt).sum(dim=(-2, -1))

        loss = -torch.mean(log_probs / num_atoms)

        return loss

