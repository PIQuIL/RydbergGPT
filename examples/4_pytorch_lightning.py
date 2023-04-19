import argparse
from dataclasses import asdict, dataclass

import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.profilers as pl_profilers
import torch
import torch.optim as optim
import torch.profiler
from config.utils import create_config_from_yaml
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    ModelCheckpoint,
    RichModelSummary,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import TensorBoardLogger

# from pytorch_lightning.profilers import AdvancedProfiler, SimpleProfiler
from pytorch_lightning.utilities.model_summary import ModelSummary

from rydberggpt.data.loading.rydberg_dataset import get_rydberg_dataloader
from rydberggpt.models.encoder_decoder_transformer import get_rydberg_encoder_decoder
from rydberggpt.training.loss import KLLoss
from rydberggpt.utils import to_one_hot


class RydbergGPTTrainer(pl.LightningModule):
    def __init__(
        self,
        model,
        config: dataclass,
        logger: TensorBoardLogger,
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
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        cond, measurements = batch
        cond = cond.unsqueeze(1)  # [batch_size, 1, 4]
        measurements = to_one_hot(measurements, self.config.num_states)

        cond_log_probs = self.forward(measurements, cond)
        loss = self.criterion(cond_log_probs, measurements)

        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer_class = getattr(optim, self.config.optimizer)
        optimizer = optimizer_class(
            self.model.parameters(), lr=self.config.learning_rate
        )
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


def main(config_path: str):
    config = create_config_from_yaml(yaml_path=config_path)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device

    train_loader, val_loader = get_rydberg_dataloader(config.batch_size)

    model = get_rydberg_encoder_decoder(config)

    logger = TensorBoardLogger(save_dir="logs")
    rydberg_gpt_trainer = RydbergGPTTrainer(model, config, logger=logger)
    # Create a ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # Metric to monitor, e.g., validation loss
        mode="min",  # Save the model with the minimum validation loss
        save_top_k=3,  # Save the top 3 best models based on the monitored metric
        save_last=True,  # Save the last model
        verbose=True,  # Log when a new checkpoint is saved
        dirpath=f"logs/lightning_logs/version_{logger.version}/checkpoints",  # Save checkpoints in the logger's version directory
        filename="best-checkpoint-{epoch}-{val_loss:.2f}",  # Checkpoint filename format
    )

    callbacks = [
        checkpoint_callback,
        DeviceStatsMonitor(),
        StochasticWeightAveraging(config.learning_rate),
    ]

    # https://lightning.ai/docs/pytorch/stable/common/trainer.html
    profiler_class = getattr(pl_profilers, config.profiler)
    profiler = profiler_class(
        dirpath=f"logs/lightning_logs/version_{logger.version}",
        filename="performance_logs",
    )

    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        callbacks=callbacks,
        devices="auto",
        accelerator=device,
        logger=logger,
        profiler=profiler,
        enable_progress_bar=config.prog_bar,
        precision=config.precision,
        # strategy=config.strategy,
    )

    trainer.fit(rydberg_gpt_trainer, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run deep learning model with specified config."
    )
    parser.add_argument(
        "config_file_name",
        nargs="?",
        default="small",
        help="Name of the configuration file without the .yaml extension. (default: small)",
    )

    args = parser.parse_args()

    config_name = args.config_file_name
    yaml_path = f"examples/config/models/{config_name}.yaml"

    main(config_path=yaml_path)
