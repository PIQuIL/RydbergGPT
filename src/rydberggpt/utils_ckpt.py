import os
import re
from typing import Any, Dict, List, Tuple, Type, Union

import pytorch_lightning as pl
from torch import nn

from rydberggpt.training.trainer import RydbergGPTTrainer
from rydberggpt.utils import create_config_from_yaml, load_yaml_file


def get_ckpt_path(from_ckpt: int, log_dir: str = "logs\\lightning_logs") -> str:
    """
    Get the checkpoint path from a specified checkpoint version number.

    Args:
        from_ckpt (int): The version number of the checkpoint.
        log_dir (str, optional): The root directory where checkpoints are stored.
                                 Defaults to "logs/lightning_logs".

    Returns:
        str: The path to the specified checkpoint version directory.

    Raises:
        FileNotFoundError: If no checkpoint is found in the specified directory.
    """
    log_dir = os.path.join(log_dir, f"version_{from_ckpt}")

    if log_dir is None:
        raise FileNotFoundError(f"No checkpoint found in {log_dir}")

    return log_dir


def find_latest_ckpt(log_dir: str):
    """
    Find the latest checkpoint file (based on modification time) in the specified log directory.

    Args:
        log_dir (str): The path to the log directory containing the checkpoint files.

    Returns:
        str: The path to the latest checkpoint file.
    """
    log_dir = os.path.join(log_dir, "checkpoints")
    ckpt_files = [file for file in os.listdir(log_dir) if file.endswith(".ckpt")]

    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")

    ckpt_files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)))
    latest_ckpt = ckpt_files[-1]
    return os.path.join(log_dir, latest_ckpt)


def find_best_ckpt(log_dir: str) -> Union[str, None]:
    """
    Find the best checkpoint file (with the lowest training loss) in the specified log directory.

    Args:
        log_dir (str): The path to the log directory containing the checkpoint files.

    Returns:
        str: The path to the checkpoint file with the lowest training loss.
    """
    log_dir = os.path.join(log_dir, "checkpoints")
    ckpt_files = [file for file in os.listdir(log_dir) if file.endswith(".ckpt")]

    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")

    # Extract the training loss from the ckpt filenames
    ckpt_losses = []
    for file in ckpt_files:
        match = re.search(r"train_loss=(\d+\.\d+)", file)
        if match:
            ckpt_losses.append(float(match.group(1)))
        else:
            ckpt_losses.append(float("inf"))

    # Find the index of the ckpt with the lowest training loss
    best_ckpt_index = ckpt_losses.index(min(ckpt_losses))
    best_ckpt = ckpt_files[best_ckpt_index]

    return os.path.join(log_dir, best_ckpt)


def get_model_from_ckpt(
    log_path: str,
    model: nn.Module,
    ckpt: str = "best",
    trainer: Type[pl.LightningModule] = RydbergGPTTrainer,
) -> nn.Module:
    """
    Load a model from a specified checkpoint file in the log directory.

    Args:
        log_path (str): The path to the log directory containing the checkpoint files.
        model (nn.Module): The model class to load.
        ckpt (str, optional): The checkpoint to load. Must be either "best" or "latest". Defaults to "best".
        trainer (Type[pl.LightningModule], optional): The trainer class to use for loading the model. Defaults to RydbergGPTTrainer.

    Returns:
        nn.Module: The loaded model.

    Raises:
        ValueError: If the value of ckpt is not "best" or "latest".
    """
    if ckpt == "best":
        ckpt_path = find_best_ckpt(log_path)
    elif ckpt == "last":
        ckpt_path = find_latest_ckpt(log_path)
    else:
        raise ValueError(f"ckpt must be 'best' or 'latest', not {ckpt}")

    yaml_dict = load_yaml_file(log_path, "hparams.yaml")
    config = create_config_from_yaml(yaml_dict)

    rydberg_gpt_trainer = trainer.load_from_checkpoint(
        ckpt_path,
        model=model,
        config=config,
        logger=None,
        example_input_array=None,
    )
    return rydberg_gpt_trainer.model
