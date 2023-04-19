import copy
from dataclasses import dataclass

import torch
import torch.optim as optim
from einops import rearrange

from rydberggpt.data.loading.rydberg_dataset import get_rydberg_dataloader
from rydberggpt.models.decoder_transformer import get_rydberg_decoder
from rydberggpt.models.encoder_decoder_transformer_2 import get_rydberg_encoder_decoder_2
from rydberggpt.training.loss import LabelSmoothing
from rydberggpt.utils import to_one_hot


@dataclass
class Config:
    # transformer
    num_heads: int = 8
    d_model: int = 32
    num_blocks: int = 2
    d_ff = 4 * d_model
    dropout = 0.0
    # training
    num_epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 0.01
    # dataset
    num_atoms: int = None
    num_samples: int = None
    delta: float = None
    # rydberg
    num_states: int = 2
    num_encoder_embedding_dims: int = 4
    device: str = None


# seed everything
torch.manual_seed(0)
config = Config()

#### MODEL
# model = get_rydberg_encoder_decoder_2(config)
model = get_rydberg_decoder(config)


train_loader, test_loader = get_rydberg_dataloader(config.batch_size, test_size=0.01)
criterion = LabelSmoothing(0.1)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)


for epoch in range(config.num_epochs):
    # set the model to train mode
    model.train()
    # loop over the training data in batches
    for i, batch in enumerate(train_loader):
        condition, measurements = batch
        condition = rearrange(condition, "b c -> b 1 c")
        measurements = to_one_hot(measurements, config.num_states)
        optimizer.zero_grad()
        # out = model.forward(condition, measurements)
        out = model.forward(measurements)
        log_cond_probs = model.generator(out)
        loss = criterion(log_cond_probs, measurements)
        print(loss, "loss")
        assert not torch.isnan(loss), "Loss is NaN"
        loss.backward()
        optimizer.step()
