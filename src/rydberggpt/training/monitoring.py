import pytorch_lightning.profilers as pl_profilers
import torch


def setup_profiler(config, log_path):
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
    return profiler
