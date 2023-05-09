import os
import re


def get_ckpt_path(from_ckpt: int, log_dir: str = "logs/lightning_logs"):
    log_dir = os.path.join(log_dir, f"version_{from_ckpt}")
    # ckpt_path = find_latest_ckpt(from_ckpt, log_dir)
    if log_dir is None:
        raise FileNotFoundError(f"No checkpoint found in {log_dir}")
    return log_dir


def set_example_input_array(train_loader):
    example_batch = next(iter(train_loader))
    return example_batch.m_onehot, example_batch.graph
