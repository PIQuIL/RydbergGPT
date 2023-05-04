import os


def find_latest_checkpoint(from_checkpoint: int, log_dir: str = "logs/lightning_logs"):
    log_dir = os.path.join(log_dir, f"version_{from_checkpoint}/checkpoints")
    checkpoint_files = [file for file in os.listdir(log_dir) if file.endswith(".ckpt")]

    if not checkpoint_files:
        return None

    checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)))
    latest_checkpoint = checkpoint_files[-1]
    return os.path.join(log_dir, latest_checkpoint)


def set_example_input_array(train_loader):
    example_batch = next(iter(train_loader))
    return example_batch.m_onehot, example_batch.graph
