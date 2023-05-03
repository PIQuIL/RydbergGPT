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
        cond_log_probs = self.forward(batch.m_shifted_onehot, batch.graph)
        loss = self.criterion(cond_log_probs, batch.m_onehot)
        self.log("train_loss", loss, sync_dist=True)
        return loss

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

    def configure_optimizers(self):
        optimizer_class = getattr(optim, self.config.optimizer)
        optimizer = optimizer_class(
            self.model.parameters(), lr=self.config.learning_rate
        )

        # Add learning rate scheduler
        # https://pytorch.org/docs/stable/optim.html
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
