import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, cond_log_probs, tgt):
        """Compute the cross-entropy loss with label smoothing.

        Args:
            cond_probs (torch.Tensor): Tensor of shape (batch_size, num_atoms, dim) containing the
                predicted conditional probabilities for each class.
            tgt (torch.Tensor): Tensor of shape (batch_size, num_atoms) containing the target labels.
                possible measurement outcomes are [0, 1].

        Returns:
            torch.Tensor: Scalar tensor representing the loss with label smoothing.
        """
        # TODO implement label smoothing
        batchsize = tgt.shape[0]
        log_probs = torch.einsum("bnd,bnd->b", cond_log_probs, tgt).sum(-1)
        loss = -torch.sum(log_probs) / batchsize
        return loss
