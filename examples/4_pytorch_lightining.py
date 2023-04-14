from dataclasses import dataclass

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim

# from lightning.pytorch import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn, optim
from torch.utils.data import DataLoader

# seed_everything(42, workers=True)
from rydberggpt.data.loading.dataset_rydberg import get_dataloaders, load_dataset
from rydberggpt.models.rydberg_transformer import get_rydberg_transformer
from rydberggpt.models.transformer.layers import DecoderLayer, EncoderLayer
from rydberggpt.models.transformer.loss import LabelSmoothing
from rydberggpt.models.transformer.models import Decoder, Encoder, Generator


@dataclass
class Config:
    # transformer
    num_heads: int = 8
    d_model: int = 32
    num_blocks: int = 2
    d_ff = 4 * d_model
    dropout = 0.0
    # training
    num_epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 0.01
    # dataset
    num_atoms: int = None
    num_samples: int = None
    delta: float = None
    # rydberg
    num_states: int = 2
    num_encoder_embedding_dims: int = 4


class RydbergGPTTrainer(pl.LightningModule):
    def __init__(
        self,
        model,
        data,
        config: Config,
    ):
        super().__init__()
        self.model = model
        self.data = data
        self.config = config

        self.criterion = LabelSmoothing(0.0)

    def forward(self, inputs, cond):
        out = self.model.forward(inputs, cond)
        cond_log_probs = self.model.generator(out)
        return cond_log_probs

    def training_step(self, batch, batch_idx):
        inputs, cond = batch
        inputs = nn.functional.one_hot(inputs, 2)
        inputs = inputs.to(torch.float)

        cond_log_probs = self.forward(inputs, cond)
        loss = self.criterion(cond_log_probs, inputs)
        assert not torch.isnan(loss), "Loss is NaN"
        self.log("train_loss", loss, prog_bar=True)
        return loss  # add this line

    def validation_step(self, batch, batch_idx):
        inputs, cond = batch
        inputs = nn.functional.one_hot(inputs, 2)
        inputs = inputs.to(torch.float)

        cond_log_probs = self.forward(inputs, cond)
        loss = self.criterion(cond_log_probs, inputs)
        self.log("val_loss", loss, prog_bar=True)
        return loss  # add this line

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        return optimizer

    def train_dataloader(self):
        train_loader, _, _ = get_dataloaders(self.data, self.config)
        return train_loader

    def val_dataloader(self):
        _, val_loader, _ = get_dataloaders(self.data, self.config)
        return val_loader


seed = 42
# seed everything
torch.manual_seed(seed)
np.random.seed(seed)


# LOAD DATA
data, dataset_config = load_dataset(delta_id=0)

# create model and train configs
config = Config(
    num_atoms=dataset_config.num_atoms,
    num_samples=dataset_config.num_samples,
    delta=dataset_config.delta,
)

model = get_rydberg_transformer(config)

# create the trainer class
rydberg_gpt_trainer = RydbergGPTTrainer(model, data, config)
logger = TensorBoardLogger(save_dir="logs", log_graph=True)

# https://lightning.ai/docs/pytorch/stable/common/trainer.html
device = "cuda" if torch.cuda.is_available() else "cpu"
trainer = pl.Trainer(
    max_epochs=config.num_epochs,
    devices="auto",
    accelerator=device,
    logger=logger,
)

trainer.fit(rydberg_gpt_trainer)
