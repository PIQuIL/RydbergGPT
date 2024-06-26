import argparse
import logging

import numpy as np
import torch
import torch.profiler
from pytorch_lightning.loggers import TensorBoardLogger
from rydberggpt.training.logger import setup_logger
from rydberggpt.training.train import train
from rydberggpt.utils import create_config_from_yaml, load_yaml_file

torch.set_float32_matmul_precision("medium")


def setup_environment(config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device


def load_configuration(config_path: str, config_name: str):
    yaml_dict = load_yaml_file(config_path, config_name)
    return create_config_from_yaml(yaml_dict)


def main(config_path: str, config_name: str, dataset_path: str):
    config = load_configuration(config_path, config_name)
    setup_environment(config)

    tensorboard_logger = TensorBoardLogger(save_dir="logs")
    tensorboard_logger.log_hyperparams(vars(config))

    log_path = f"logs/lightning_logs/version_{tensorboard_logger.version}"
    logging.info(f"Log path: {log_path}")

    setup_logger(log_path)
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logging.info(f"Found {num_gpus} GPUs.")
    else:
        logging.info("No GPUs found.")

    train(config, dataset_path, tensorboard_logger, log_path)


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
        "--config_path",
        default="config/",
        help="Path to the configuration file. (default: config/)",
    )
    parser.add_argument(
        "--dataset_path",
        default="dataset_test/",
        help="Name of the configuration file without the .yaml extension. (default: small)",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=0,
        help="Number of devices (GPUs) to use. (default: 0)",
    )

    args = parser.parse_args()

    main(
        config_path=args.config_path,
        config_name=args.config_name,
        dataset_path=args.dataset_path,
    )
