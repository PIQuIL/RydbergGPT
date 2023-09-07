import argparse
from dataclasses import asdict, dataclass
from typing import Dict, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.profiler
from pytorch_lightning.loggers import TensorBoardLogger
from torch import optim

from rydberggpt.training import loss


class RydbergGPTTrainer(pl.LightningModule):
    """
    A custom PyTorch Lightning module for training a Rydberg GPT model.

    Args:
        model (nn.Module): The model to be trained.
        config (dataclass): A dataclass containing the model's configuration parameters.
        logger (TensorBoardLogger): A TensorBoard logger instance for logging training progress.
        example_input_array (torch.tensor, optional): An example input tensor used for
            generating the model summary.
    """

    def __init__(
        self,
        model: nn.Module,
        config: dataclass,
        logger: TensorBoardLogger = None,
        example_input_array: torch.tensor = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters(asdict(config))
        self.model = model
        self.criterion = getattr(loss, self.config.criterion)()
        self.example_input_array = example_input_array

    def forward(self, m_onehot: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the model.

        Args:
            m_onehot (torch.Tensor): One-hot encoded measurements tensor.
            cond (torch.Tensor): Conditioning tensor. # TODO prompt

        Returns:
            torch.Tensor: Conditional log probabilities tensor.
        """
        out = self.model.forward(m_onehot, cond)
        cond_log_probs = self.model.generator(out)
        return cond_log_probs

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Perform a single training step.

        Args:
            batch (pl.Batch): A batch of data during training.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The training loss for the current batch.
        """
        cond_log_probs = self.forward(batch.m_shifted_onehot, batch.graph)
        loss = self.criterion(cond_log_probs, batch.m_onehot)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self) -> Dict[str, Union[optim.Optimizer, Dict]]:
        """
        Configures the optimizer and learning rate scheduler for the RydbergGPTTrainer.

        Returns:
            Dict[str, Union[optim.Optimizer, Dict]]: A dictionary containing the optimizer and lr_scheduler configurations.
        """
        optimizer_class = getattr(optim, self.config.optimizer)
        optimizer = optimizer_class(
            self.model.parameters(), lr=self.config.learning_rate
        )

        # Add learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.config.t_initial,  # initial number of epochs in a period
            T_mult=self.config.t_mult,  # factor to increase the period length after each restart
            eta_min=self.config.eta_min,  # minimum learning rate
        )

        # Return both the optimizer and the scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "train_loss",
            },
        }

    # def validation_step(self, batch, batch_idx):  # batch denotes the batch
    #     self.model.eval()
    #     # TODO add calculation of energy function as well
    #     # we can track the energy of the dataset vs the energy of the model
    #     batch_size, num_atoms, _ = batch.m_onehot.shape
    #     m = torch.argmax(batch.m_onehot, dim=-1)
    #     samples, log_probs = self.model.get_samples_and_log_probs(
    #         batch_size, batch.cond, num_atoms, device=self.device
    #     )
    #     # log_probs = cond_log_probs.sum(dim=-1)
    #     energy = self.model.get_energy(
    #         V=batch.coupling_matrix,
    #         omega=batch.omega,
    #         delta=batch.delta,
    #         samples=samples,
    #         cond=batch.cond,
    #         log_probs=log_probs,
    #         num_atoms=num_atoms,
    #         device=self.device,
    #     )
    #     # cond_log_probs = self.forward(batch.m_shifted_onehot, batch.cond)
    #     # loss = self.criterion(cond_log_probs, batch.m_onehot)
    #     self.log("energy", energy, sync_dist=True)
    #     return energy
