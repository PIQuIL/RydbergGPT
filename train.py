import argparse
from typing import Optional

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
from rydberggpt.data.loading.rydberg_dataset_chunked import get_chunked_dataloader
from rydberggpt.models.rydberg_encoder_decoder import get_rydberg_graph_encoder_decoder
from rydberggpt.training.callbacks.module_info_callback import ModelInfoCallback
from rydberggpt.training.callbacks.stop_on_loss_threshold_callback import (
    StopOnLossThreshold,
)
from rydberggpt.training.trainer import RydbergGPTTrainer
from rydberggpt.training.utils import set_example_input_array
from rydberggpt.utils import create_config_from_yaml, load_yaml_file
from rydberggpt.utils_ckpt import (
    find_best_ckpt,
    find_latest_ckpt,
    get_ckpt_path,
    get_model_from_ckpt,
)

torch.set_float32_matmul_precision("medium")


def main(config_path: str, config_name: str, dataset_path: str):
    yaml_dict = load_yaml_file(config_path, config_name)
    config = create_config_from_yaml(yaml_dict)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device

    # https://lightning.ai/docs/pytorch/stable/data/datamodule.html
    train_loader, val_loader = get_chunked_dataloader(
        config.batch_size,
        test_size=0.2,
        num_workers=config.num_workers,
        data_path=dataset_path,
    )
    input_array = set_example_input_array(train_loader)

    model = get_rydberg_graph_encoder_decoder(config)

    # Compile model
    if config.compile:
        # check that device is cuda
        if device != "cuda":
            raise ValueError(
                "Cannot compile model if device is not cuda. "
                "Please set compile to False."
            )
        model = torch.compile(model)

    # Setup tensorboard logger
    logger = TensorBoardLogger(save_dir="logs")
    log_path = f"logs/lightning_logs/version_{logger.version}"
    rydberg_gpt_trainer = RydbergGPTTrainer(
        model, config, logger=logger  # , example_input_array=input_array
    )

    # Callbacks
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
        # StopOnLossThreshold(loss_threshold=150.0),
    ]

    # Monitoring
    if config.advanced_monitoring:
        # https://lightning.ai/docs/pytorch/stable/common/trainer.html
        profiler = getattr(pl_profilers, config.profiler)(
            schedule=torch.profiler.schedule(
                skip_first=1, wait=1, warmup=1, active=3, repeat=1000
            ),  # everything is happening in steps
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_path),
            with_stack=True,
            with_modules=True,
            profile_memory=True,
            with_flops=True,
            record_shapes=True,
            dirpath=log_path,
            filename="performance_logs",
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],  # records both CPU and CUDA activities
        )
    else:
        # https://lightning.ai/docs/pytorch/stable/common/trainer.html
        profiler_class = getattr(pl_profilers, config.profiler)
        profiler = profiler_class(
            dirpath=log_path,
            filename="performance_logs",
        )

    # Distributed training
    if config.strategy == "ddp":
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = config.strategy

    # Init trainer class
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
        log_every_n_steps=config.log_every,
        # overfit_batches=1,
        accumulate_grad_batches=config.accumulate_grad_batches,
        detect_anomaly=config.detect_anomaly,
    )

    # Find the latest checkpoint
    if config.from_checkpoint is not None:
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
        default="data",
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

    print(args.dataset_path)

    main(config_path=yaml_path, config_name=config_name, dataset_path=args.dataset_path)
