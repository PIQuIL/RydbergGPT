import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class KLLoss(nn.Module):
    """
    KLLoss is a custom loss function class that computes the Kullback-Leibler divergence
    between the target distribution (tgt) and the conditional log probabilities
    (cond_log_probs).
    """

    def __init__(self):
        super(KLLoss, self).__init__()

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
        batchsize = tgt.shape[0]
        # cond_probs = torch.exp(cond_log_probs)  # used for debugging
        temp = torch.einsum("bnd,bnd->bn", cond_log_probs, tgt)
        log_probs = torch.sum(temp, axis=-1)
        loss = -torch.sum(log_probs) / batchsize
        return loss


class LabelSmoothing(nn.Module):
    """Implement label smoothing for a classification task. Label smoothing is a regularization
    technique that smooths the probability distribution of the target labels by replacing the
    hard 0s and 1s in the one-hot target vectors with small positive values.

    https://arxiv.org/abs/1906.02629

    Args:
        smoothing (float, optional): Smoothing factor to apply. Defaults to 0.0.
    """

    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.smoothing = smoothing

    def forward(self, cond_log_probs: Tensor, tgt: Tensor) -> Tensor:
        """Compute the cross-entropy loss with label smoothing.

        Args:
            cond_probs (torch.Tensor): Tensor of shape (batch_size, num_atoms, dim) containing the
                predicted conditional probabilities for each class.
            tgt (torch.Tensor): Tensor of shape (batch_size, num_atoms, 2) containing the target labels.
                possible measurement outcomes one-hot encoded [0, 1] or [1, 0].

        Returns:
            torch.Tensor: Scalar tensor representing the loss with label smoothing.
        """
        # TODO implement label smoothing
        batchsize = tgt.shape[0]
        # cond_probs = torch.exp(cond_log_probs)  # used for debugging
        temp = torch.einsum("bnd,bnd->bn", cond_log_probs, tgt)
        log_probs = torch.sum(temp, axis=-1)
        loss = -torch.sum(log_probs) / batchsize
        return loss
