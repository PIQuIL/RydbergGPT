import pytorch_lightning as pl
import torch
from torch import Tensor


class NLLLoss(pl.LightningModule):
    """
    This class implements the Negative Log Likelihood (NLL) loss function as a PyTorch Lightning module.

    The NLL loss measures the performance of a classification model where the prediction input is a probability
    distribution over classes. It is useful in training models for multi-class classification problems.

    The loss is calculated by taking the negative log of the probabilities predicted by the model for the true class labels.

    Methods:
        forward(cond_log_probs: Tensor, tgt: Tensor) -> Tensor:
            Computes the NLL loss based on the conditional log probabilities and the target values.

    Examples:
        >>> nll_loss = NLLLoss()
        >>> loss = nll_loss(cond_log_probs, tgt)
    """

    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, cond_log_probs: Tensor, tgt: Tensor) -> Tensor:
        """
        Computes the NLL loss based on the conditional log probabilities and the target values.

        Args:
            cond_log_probs (Tensor): The conditional log probabilities predicted by the model.
            tgt (Tensor): The target values.

        Returns:
            Tensor: The computed NLL loss.
        """
        num_atoms = tgt.shape[-2] - (tgt == 0.0).all(-1).sum(-1)
        log_probs = (cond_log_probs * tgt).sum(dim=(-2, -1))
        loss = -torch.mean(log_probs / num_atoms)
        return loss
