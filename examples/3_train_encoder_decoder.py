import copy
from dataclasses import dataclass

import torch
import torch.optim as optim
from torch import nn
from torchsummary import summary

# from rydberggpt.data.loading.dataset_rydberg import get_dataloaders, load_dataset
from rydberggpt.data.loading.rydberg_dataset import get_rydberg_dataloader
from rydberggpt.models.encoder_decoder_transformer import get_rydberg_encoder_decoder
from rydberggpt.training.loss import LabelSmoothing
from rydberggpt.utils import to_one_hot

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


config = Config()
train_loader, test_loader = get_rydberg_dataloader(config.batch_size, test_size=0.01)
model = get_rydberg_encoder_decoder(config)
# model = get_rydberg_transformer_decoder_only(config)
criterion = LabelSmoothing(0.1)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)


for epoch in range(config.num_epochs):
    # set the model to train mode
    model.train()
    # loop over the training data in batches
    for i, batch in enumerate(train_loader):
        condition, measurements = batch
        condition = condition.unsqueeze(1)  # [batch_size, 1, 4]
        # condition.requires_grad_(True)
        measurements = to_one_hot(measurements, config.num_states)
        optimizer.zero_grad()
        # out = model.forward(measurements)
        out = model.forward(measurements, condition)
        log_cond_probs = model.generator(out)
        loss = criterion(log_cond_probs, measurements)
        print(loss, "loss")
        assert not torch.isnan(loss), "Loss is NaN"
        loss.backward()
        optimizer.step()


# Try getting grad from
def custom_loss_inference(out, condition, measurements):
    log_cond_probs = model.generator(out)
    base_loss = criterion(log_cond_probs, measurements)

    # Add a term to the base_loss that depends on the condition
    # For example, use the mean of the absolute values of the condition tensor
    penalty = torch.mean(torch.abs(condition))

    # Adjust the weight of the penalty term according to your needs
    penalty_weight = 0.1
    total_loss = base_loss + penalty_weight * penalty

    return total_loss


model.eval()  # Set the model to evaluation mode

for i, batch in enumerate(train_loader):
    measurements, condition = batch
    condition.requires_grad_(True)

    measurements = nn.functional.one_hot(measurements, 2)
    measurements = measurements.to(torch.float)

    with torch.no_grad():
        out = model.forward(measurements, condition)

    loss = custom_loss_inference(out, condition, measurements)
    loss.backward()
    condition_gradient = condition.grad
    print(condition_gradient, "condition grad")

    # ... use the condition_gradient for your analysis or further processing
