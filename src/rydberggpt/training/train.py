import logging

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

# from rydberggpt.data.loading.rydberg_dataset import get_rydberg_dataloader
from rydberggpt.data.loading.rydberg_dataset_2 import get_rydberg_dataloader_2
from rydberggpt.models.rydberg_encoder_decoder import get_rydberg_graph_encoder_decoder
from rydberggpt.training.callbacks.module_info_callback import ModelInfoCallback
from rydberggpt.training.callbacks.stop_on_loss_threshold_callback import (
    StopOnLossThreshold,
)
from rydberggpt.training.logger import setup_logger
from rydberggpt.training.monitoring import setup_profiler
from rydberggpt.training.trainer import RydbergGPTTrainer
from rydberggpt.training.utils import set_example_input_array
from rydberggpt.utils_ckpt import (
    find_best_ckpt,
    find_latest_ckpt,
    get_ckpt_path,
    get_model_from_ckpt,
)

torch.set_float32_matmul_precision("medium")


def load_data(config, dataset_path):
    logging.info(f"Loading data from {dataset_path}...")
    train_loader, val_loader = get_rydberg_dataloader_2(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        data_path=dataset_path,
        buffer_size=config.buffer_size,
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
        # ModelInfoCallback(),
        LearningRateMonitor(logging_interval="step"),
    ]
    return callbacks


def train(
    config: dict,
    dataset_path: str,
    tensorboard_logger: TensorBoardLogger,
    log_path: str,
):
    model = create_model(config)

    rydberg_gpt_trainer = RydbergGPTTrainer(model, config, logger=tensorboard_logger)
    callbacks = setup_callbacks(config)
    profiler = setup_profiler(config, log_path)
    strategy = (
        DDPStrategy(find_unused_parameters=True)
        if config.strategy == "ddp"
        else config.strategy
    )

    trainer = pl.Trainer(
        devices="auto",
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

    train_loader, val_loader = load_data(config, dataset_path)
    # input_array = set_example_input_array(train_loader)

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
