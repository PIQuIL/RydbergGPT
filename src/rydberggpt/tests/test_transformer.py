import pytest
import torch
from torch import nn

from rydberggpt.models.transformer_wavefunction import TransformerWavefunction
from rydberggpt.training.loss import LabelSmoothing


def get_dummy_data():
    num_atoms = 3  # number of atoms
    num_samples = 8  # number of samples

    H = torch.rand(
        (num_samples, num_atoms, 4), dtype=torch.float
    )  # [batch_size , num_atoms, 4]
    dataset = torch.randint(0, 2, (num_samples, num_atoms), dtype=torch.int64)
    return H, dataset


def test_model_minimizes_loss():
    # prepare dummy data
    H, dataset = get_dummy_data()
    inputs = nn.functional.one_hot(dataset, 2)
    inputs = inputs.to(torch.float)

    # initialize model, criterion, and optimizer
    model = TransformerWavefunction(10, 2, 2)
    criterion = LabelSmoothing()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    # calculate initial loss
    cond_probs = model.forward([H, inputs])
    first_loss = criterion(cond_probs, inputs)

    # train the model for 50 iterations
    for _ in range(50):
        optimizer.zero_grad()
        cond_probs = model.forward([H, inputs])
        loss = criterion(cond_probs, inputs)
        loss.backward()
        optimizer.step()

    # check if the final loss is smaller than the initial loss
    assert loss < first_loss, "Final loss is not smaller than initial loss"


if __name__ == "__main__":
    pytest.main([__file__])
    # test_model_minimizes_loss()
