import torch
import torch.nn as nn
import torch.nn.functional as F


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, cond_log_probs, tgt):
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

    def forward(self, cond_log_probs, tgt):
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
        # log_probs = torch.einsum("bnd,bnd->b", cond_log_probs, tgt).sum(-1)
        # log_probs = torch.sum(torch.sum(cond_log_probs * tgt, axis=-1), axis=-1)
        cond_probs = torch.exp(cond_log_probs)  # used for debugging
        temp = torch.einsum("bnd,bnd->bn", cond_log_probs, tgt)
        log_probs = torch.sum(temp, axis=-1)
        # assert torch.allclose(log_probs_test, log_probs_test_2)
        # assert torch.allclose(log_probs, torch.sum(log_probs_test_2))
        loss = -torch.sum(log_probs) / batchsize
        return loss
