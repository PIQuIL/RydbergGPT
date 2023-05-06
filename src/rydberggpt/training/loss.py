import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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
        # loss = self.criterion(cond_log_probs, tgt)
        # cond_probs = torch.exp(cond_log_probs)  # used for debugging
        # batchsize = tgt.shape[0]
        # log_probs = torch.sum(temp, axis=-1)
        # loss = -torch.sum(log_probs) / batchsize
        log_probs = torch.einsum("bnd,bnd->b", cond_log_probs, tgt)
        loss = -torch.mean(log_probs)
        return loss


class LabelSmoothing(pl.LightningModule):
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
        self.criterion = nn.KLDivLoss(reduction="batchmean")

    def forward(self, cond_log_probs: Tensor, tgt: Tensor) -> Tensor:
        """Compute the KLDiv loss with label smoothing.

        Args:
            cond_probs (torch.Tensor): Tensor of shape (batch_size, num_atoms, dim) containing the
                predicted conditional probabilities for each class.
            tgt (torch.Tensor): Tensor of shape (batch_size, num_atoms, 2) containing the target labels.
                possible measurement outcomes one-hot encoded [0, 1] or [1, 0].

        Returns:
            torch.Tensor: Scalar tensor representing the loss with label smoothing.
        """
        assert 0 <= self.smoothing < 1
        num_classes = tgt.shape[-1]

        # Compute smoothed target labels
        with torch.no_grad():
            smoothed_tgt = tgt * (1.0 - self.smoothing) + self.smoothing / num_classes

        # Compute the Kullback-Leibler divergence loss
        loss = self.criterion(cond_log_probs, smoothed_tgt)

        return loss
