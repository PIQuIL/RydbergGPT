from pytorch_lightning.callbacks import Callback


class StopOnLossThreshold(Callback):
    def __init__(self, loss_threshold: float = 1.0):
        super().__init__()
        self.loss_threshold = loss_threshold

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs["loss"]
        if loss < self.loss_threshold:
            trainer.should_stop = True
            return
