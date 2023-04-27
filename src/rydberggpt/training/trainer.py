import argparse
from dataclasses import asdict, dataclass

import pytorch_lightning as pl
import torch
import torch.optim as optim
import torch.profiler
from pytorch_lightning.loggers import TensorBoardLogger

from rydberggpt.training.loss import KLLoss


class RydbergGPTTrainer(pl.LightningModule):
    def __init__(
        self,
        model,
        config: dataclass,
        logger: TensorBoardLogger,
        example_input_array: torch.tensor = None,
    ):
        super().__init__()
        self.config = config
        self.save_hyperparameters(asdict(config))
        self.model = model
        self.criterion = KLLoss()
        self.example_input_array = example_input_array

    def forward(self, m_onehot, cond):
        out = self.model.forward(m_onehot, cond)
        cond_log_probs = self.model.generator(out)
        return cond_log_probs

    def training_step(self, batch, batch_idx):
        cond, m_shifted_onehot, m_onehot = batch
        cond_log_probs = self.forward(m_shifted_onehot, cond)
        loss = self.criterion(cond_log_probs, m_onehot)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        cond, m_shifted_onehot, m_onehot = batch
        cond_log_probs = self.forward(m_shifted_onehot, cond)
        loss = self.criterion(cond_log_probs, m_onehot)
        self.log("val_loss", loss, sync_dist=True)
        return loss

    # # TODO add learning rate scheduler
    # def configure_optimizers(self):
    #     optimizer_class = getattr(optim, self.config.optimizer)
    #     optimizer = optimizer_class(
    #         self.model.parameters(), lr=self.config.learning_rate
    #     )
    #     return optimizer

    def configure_optimizers(self):
        optimizer_class = getattr(optim, self.config.optimizer)
        optimizer = optimizer_class(
            self.model.parameters(), lr=self.config.learning_rate
        )

        # Add learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            # step_size=self.config.scheduler_step_size,
            step_size=30,
            # gamma=self.config.scheduler_gamma,
        )

        # Return both the optimizer and the scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # or 'step' if you want to update the learning rate every step
                "monitor": "val_loss",  # monitor the validation loss for updating the learning rate
            },
        }
