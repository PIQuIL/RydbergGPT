import copy
from dataclasses import dataclass

import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm

from rydberggpt.data.loading.dataset_rydberg import get_dataloaders, load_dataset
from rydberggpt.models.rydberg_transformer import get_rydberg_transformer
from rydberggpt.models.transformer.loss import LabelSmoothing

# seed everything
torch.manual_seed(0)


@dataclass
class Config:
    # transformer
    num_heads: int = 8
    d_model: int = 32
    num_blocks: int = 2
    d_ff = 4 * d_model
    dropout = 0.0
    # training
    num_epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 0.01
    # dataset
    num_atoms: int = None
    num_samples: int = None
    delta: float = None
    # rydberg
    num_states: int = 2
    num_encoder_embedding_dims: int = 4


# LOAD DATA
data, dataset_config = load_dataset(delta_id=0)

config = Config(
    num_atoms=dataset_config.num_atoms,
    num_samples=dataset_config.num_samples,
    delta=dataset_config.delta,
)

train_loader, val_loader, test_loader = get_dataloaders(data, config)


model = get_rydberg_transformer(config)

# define the loss function
criterion = LabelSmoothing(0.1)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
# H = torch.rand((config.batch_size, dataset_config.num_atoms, 4), dtype=torch.float)

# loop over the data for the specified number of epochs
for epoch in range(config.num_epochs):
    # set the model to train mode
    model.train()

    # loop over the training data in batches
    for i, batch in enumerate(train_loader):
        inputs, condition = batch
        inputs = nn.functional.one_hot(inputs, 2)
        inputs = inputs.to(torch.float)

        optimizer.zero_grad()

        out = model.forward(inputs, condition)
        log_cond_probs = model.generator(out)
        loss = criterion(log_cond_probs, inputs)
        print(loss)
        # assert loss is not Nan
        assert not torch.isnan(loss), "Loss is NaN"
        loss.backward()
        optimizer.step()
