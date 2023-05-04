import yaml
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.model_summary import ModelSummary

from rydberggpt.models.utils import extract_model_info


class ModelInfoCallback(Callback):
    """
    A custom PyTorch Lightning callback that logs model information at the start of training.

    This callback extracts and logs information about the model's structure, total parameters, and
    total trainable parameters at the beginning of the training process. The information is saved
    as a YAML file in the logger's log directory.
    """

    def on_train_start(self, trainer, pl_module) -> None:
        """
        Run the callback at the beginning of training.

        Args:
            trainer (pytorch_lightning.Trainer): The PyTorch Lightning trainer instance.
            pl_module (pytorch_lightning.LightningModule): The PyTorch Lightning module instance.
        """
        # This will run at the beginning of training
        log_path = trainer.logger.log_dir

        summary = ModelSummary(pl_module, max_depth=1)
        total_parameters = summary.total_parameters
        total_trainable_parameters = summary.trainable_parameters

        summary_dict = extract_model_info(pl_module.model)
        summary_dict["total_parameters"] = total_parameters
        summary_dict["total_trainable_parameters"] = total_trainable_parameters

        # Save the summary dictionary to a YAML file
        with open(f"{log_path}/model_info.yaml", "w") as file:
            yaml.dump(summary_dict, file)
