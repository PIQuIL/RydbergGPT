from dataclasses import asdict, dataclass

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn, optim
from torch.utils.data import DataLoader
from torchsummary import summary

# seed_everything(42, workers=True)
from rydberggpt.data.loading.rydberg_dataset import get_rydberg_dataloader
from rydberggpt.models.encoder_decoder_transformer import get_rydberg_encoder_decoder
from rydberggpt.training.loss import KLLoss, LabelSmoothing
from rydberggpt.utils import to_one_hot


@dataclass
class Config:
    # transformer
    num_heads: int = 8
    d_model: int = 32
    num_blocks: int = 2
    d_ff = 4 * d_model
    dropout = 0.1
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
    device: str = None


class RydbergGPTTrainer(pl.LightningModule):
    def __init__(
        self,
        model,
        config: Config,
    ):
        super().__init__()
        self.config = config
        self.save_hyperparameters(asdict(config))
        self.model = model
        self.criterion = KLLoss()

    def forward(self, measurements, cond):
        out = self.model.forward(measurements, cond)
        cond_log_probs = self.model.generator(out)
        return cond_log_probs

    def training_step(self, batch, batch_idx):
        cond, measurements = batch
        cond = cond.unsqueeze(1)  # [batch_size, 1, 4]
        measurements = to_one_hot(measurements, self.config.num_states)

        cond_log_probs = self.forward(measurements, cond)
        loss = self.criterion(cond_log_probs, measurements)
        assert not torch.isnan(loss), "Loss is NaN"
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        cond, measurements = batch
        cond = cond.unsqueeze(1)  # [batch_size, 1, 4]
        measurements = to_one_hot(measurements, self.config.num_states)

        cond_log_probs = self.forward(measurements, cond)
        loss = self.criterion(cond_log_probs, measurements)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        return optimizer

    # def set_example_input_array(self, data, config):
    #     train_loader, _, _ = get_dataloaders(data, config)
    #     example_batch = next(iter(train_loader))
    #     measurements, cond = example_batch
    #     measurements = nn.functional.one_hot(measurements, 2)
    #     measurements = measurements.to(torch.float)
    #     # print(measurements.shape)
    #     # print(cond.shape)
    #     self.example_input_array = (measurements, cond)


seed = 42
# seed everything
torch.manual_seed(seed)
np.random.seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

# create model and train configs
config = Config(
    device=device,
)

train_loader, val_loader = get_rydberg_dataloader(config.batch_size)
# train_loader, val_loader, _ = get_dataloaders(data, config)

model = get_rydberg_encoder_decoder(config)

# Create a ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",  # Metric to monitor, e.g., validation loss
    mode="min",  # Save the model with the minimum validation loss
    save_top_k=3,  # Save the top 3 best models based on the monitored metric
    save_last=True,  # Save the last model
    verbose=True,  # Log when a new checkpoint is saved
    dirpath="checkpoints",  # Directory to save the checkpoints
    filename="best-checkpoint-{epoch}-{val_loss:.2f}",  # Checkpoint filename format
)

# create the trainer class
rydberg_gpt_trainer = RydbergGPTTrainer(model, config)
# rydberg_gpt_trainer.set_example_input_array(data, config)

# logger = TensorBoardLogger(save_dir="logs", log_graph=True)
logger = TensorBoardLogger(save_dir="logs")

# https://lightning.ai/docs/pytorch/stable/common/trainer.html
trainer = pl.Trainer(
    max_epochs=config.num_epochs,
    callbacks=[checkpoint_callback],  # Add the ModelCheckpoint callback
    devices="auto",
    accelerator=device,
    logger=logger,
)

trainer.fit(rydberg_gpt_trainer, train_loader, val_loader)
