import argparse

import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.profilers as pl_profilers
import torch
import torch.profiler
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from rydberggpt.data.loading.rydberg_dataset import get_rydberg_dataloader
from rydberggpt.models.encoder_decoder_transformer import get_rydberg_encoder_decoder
from rydberggpt.training.callbacks.module_info_callback import ModelInfoCallback
from rydberggpt.training.trainer import RydbergGPTTrainer
from rydberggpt.utils import create_config_from_yaml


def set_example_input_array(train_loader):
    example_batch = next(iter(train_loader))
    cond, m_shifted_onehot, m_onehot = example_batch
    return m_onehot, cond


def main(config_path: str):
    config = create_config_from_yaml(yaml_path=config_path)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device

    # https://lightning.ai/docs/pytorch/stable/data/datamodule.html
    train_loader, val_loader = get_rydberg_dataloader(
        config.batch_size, test_size=0.2, num_workers=config.num_workers
    )
    input_array = set_example_input_array(train_loader)

    # TODO start from checkpoint if specified
    model = get_rydberg_encoder_decoder(config)
    if config.compile:
        # check that device is cuda
        if device != "cuda":
            raise ValueError(
                "Cannot compile model if device is not cuda. "
                "Please set compile to False."
            )
        model = torch.compile(model)

    logger = TensorBoardLogger(save_dir="logs")
    log_path = f"logs/lightning_logs/version_{logger.version}"
    rydberg_gpt_trainer = RydbergGPTTrainer(
        model, config, logger=logger, example_input_array=input_array
    )

    callbacks = [
        ModelCheckpoint(
            monitor="train_loss",
            save_top_k=3,
            save_last=True,
            filename="best-checkpoint-{epoch}-{train_loss:.2f}",
        ),
        DeviceStatsMonitor(),
        StochasticWeightAveraging(config.learning_rate),
        ModelInfoCallback(),
        LearningRateMonitor(logging_interval="step"),
    ]

    # https://lightning.ai/docs/pytorch/stable/common/trainer.html
    profiler_class = getattr(pl_profilers, config.profiler)
    profiler = profiler_class(
        dirpath=log_path,
        filename="performance_logs",
    )

    if config.strategy == "ddp":
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = config.strategy

    trainer = pl.Trainer(
        devices=-1,
        strategy=strategy,
        accelerator="auto",
        precision=config.precision,
        max_epochs=config.max_epochs,
        callbacks=callbacks,
        logger=logger,
        profiler=profiler,
        enable_progress_bar=config.prog_bar,
    )

    trainer.fit(rydberg_gpt_trainer, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run deep learning model with specified config."
    )
    parser.add_argument(
        "--config_name",
        default="small",
        help="Name of the configuration file without the .yaml extension. (default: small)",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=0,
        help="Number of devices (GPUs) to use. (default: 0)",
    )

    args = parser.parse_args()

    config_name = args.config_name
    yaml_path = f"examples/config/models/{config_name}.yaml"

    main(config_path=yaml_path)
