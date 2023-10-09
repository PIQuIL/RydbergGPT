import argparse
import logging
import os
from typing import Optional

import numpy as np
import pytorch_lightning as pl
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

from rydberggpt.data.loading import (
    get_chunked_dataloader,
    get_chunked_random_dataloader,
    get_rydberg_dataloader,
    get_streaming_dataloader,
)
from rydberggpt.models.rydberg_encoder_decoder import get_rydberg_graph_encoder_decoder
from rydberggpt.training.callbacks.module_info_callback import ModelInfoCallback
from rydberggpt.training.callbacks.stop_on_loss_threshold_callback import (
    StopOnLossThreshold,
)
from rydberggpt.training.logger import setup_logger
from rydberggpt.training.monitoring import setup_profiler
from rydberggpt.training.trainer import RydbergGPTTrainer
from rydberggpt.training.utils import set_example_input_array
from rydberggpt.utils import create_config_from_yaml, load_yaml_file, save_to_yaml
from rydberggpt.utils_ckpt import (
    find_best_ckpt,
    find_latest_ckpt,
    get_ckpt_path,
    get_model_from_ckpt,
)

torch.set_float32_matmul_precision("medium")


def setup_environment(config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device


def load_data(config, dataset_path):
    logging.info(f"Loading data from {dataset_path}...")
    # https://lightning.ai/docs/pytorch/stable/data/datamodule.html
    # train_loader, val_loader = get_chunked_dataloader(
    # train_loader, val_loader = get_rydberg_dataloader(
    # train_loader, val_loader = get_streaming_dataloader(
    train_loader, val_loader = get_chunked_random_dataloader(
        config.batch_size,
        test_size=0.2,
        num_workers=config.num_workers,
        data_path=dataset_path,
        chunks_in_memory=config.chunks_in_memory,
    )
    return train_loader, val_loader


def create_model(config):
    model = get_rydberg_graph_encoder_decoder(config)
    if config.compile:
        if config.device != "cuda":
            raise ValueError(
                "Cannot compile model if device is not cuda. "
                "Please set compile to False."
            )
        model = torch.compile(model)
    return model


def setup_callbacks(config):
    logging.info("Setting up callbacks...")
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
    return callbacks


def load_configuration(config_path: str, config_name: str):
    yaml_dict = load_yaml_file(config_path, config_name)
    return create_config_from_yaml(yaml_dict)


def main(config_path: str, config_name: str, dataset_path: str):
    config = load_configuration(config_path, config_name)
    setup_environment(config)
    model = create_model(config)

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
    
    config.num_workers = num_gpus * config.num_workers_per_gpu

    tensorboard_logger = TensorBoardLogger(save_dir="logs")
    log_path = f"logs/lightning_logs/version_{tensorboard_logger.version}"
    logging.info(f"Log path: {log_path}")
    setup_logger(log_path)
    tensorboard_logger.log_hyperparams(vars(config))

    rydberg_gpt_trainer = RydbergGPTTrainer(model, config, logger=tensorboard_logger)
    callbacks = setup_callbacks(config)
    profiler = setup_profiler(config, log_path)
    strategy = (
        DDPStrategy(find_unused_parameters=True)
        if config.strategy == "ddp"
        else config.strategy
    )
    # Init trainer class
    trainer = pl.Trainer(
        devices="auto",  # check if "auto" works
        strategy=strategy,
        accelerator="auto",
        precision=config.precision,
        max_epochs=config.max_epochs,
        callbacks=callbacks,
        logger=tensorboard_logger,
        profiler=profiler,
        enable_progress_bar=config.prog_bar,
        log_every_n_steps=config.log_every,
        # overfit_batches=1,
        accumulate_grad_batches=config.accumulate_grad_batches,
        detect_anomaly=config.detect_anomaly,
    )

    # Load data
    train_loader, val_loader = load_data(config, dataset_path)
    input_array = set_example_input_array(train_loader)

    # Find the latest checkpoint
    if config.from_checkpoint is not None:
        logging.info(f"Loading model from checkpoint {config.from_checkpoint}...")
        log_path = get_ckpt_path(from_ckpt=config.from_checkpoint)
        checkpoint_path = find_latest_ckpt(log_path)
        trainer.fit(
            rydberg_gpt_trainer, train_loader, val_loader, ckpt_path=checkpoint_path
        )
    else:
        trainer.fit(rydberg_gpt_trainer, train_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run deep learning model with specified config."
    )
    parser.add_argument(
        "--config_name",
        default="config_small",
        help="Name of the configuration file without the .yaml extension. (default: small)",
    )
    parser.add_argument(
        "--dataset_path",
        default="data_old_chunked/",
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
    yaml_path = f"config/"

    main(config_path=yaml_path, config_name=config_name, dataset_path=args.dataset_path)
