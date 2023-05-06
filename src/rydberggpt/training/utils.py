import os
import re


def find_latest_checkpoint(from_checkpoint: int, log_dir: str = "logs/lightning_logs"):
    log_dir = os.path.join(log_dir, f"version_{from_checkpoint}/checkpoints")
    checkpoint_files = [file for file in os.listdir(log_dir) if file.endswith(".ckpt")]

    if not checkpoint_files:
        return None

    checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)))
    latest_checkpoint = checkpoint_files[-1]
    return os.path.join(log_dir, latest_checkpoint)


def find_best_checkpoint(from_checkpoint: int, log_dir: str = "logs/lightning_logs"):
    log_dir = os.path.join(log_dir, f"version_{from_checkpoint}/checkpoints")
    checkpoint_files = [file for file in os.listdir(log_dir) if file.endswith(".ckpt")]

    if not checkpoint_files:
        return None

    # Extract the training loss from the checkpoint filenames
    checkpoint_losses = []
    for file in checkpoint_files:
        match = re.search(r"train_loss-(\d+\.\d+)", file)
        if match:
            checkpoint_losses.append(float(match.group(1)))
        else:
            checkpoint_losses.append(float("inf"))

    # Find the index of the checkpoint with the lowest training loss
    best_checkpoint_index = checkpoint_losses.index(min(checkpoint_losses))
    best_checkpoint = checkpoint_files[best_checkpoint_index]

    return os.path.join(log_dir, best_checkpoint)


def set_example_input_array(train_loader):
    example_batch = next(iter(train_loader))
    return example_batch.m_onehot, example_batch.graph
