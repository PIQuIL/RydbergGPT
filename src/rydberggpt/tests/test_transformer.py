import pytest
import torch

from rydberggpt.models import TransformerWavefunction
from rydberggpt.models.transformer.loss import LabelSmoothing


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

    # initialize model, criterion, and optimizer
    model = TransformerWavefunction(10, 2, 2)
    criterion = LabelSmoothing()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    # calculate initial loss
    cond_probs = model.forward([H, dataset])
    first_loss = criterion(cond_probs, dataset)

    # train the model for 50 iterations
    for _ in range(50):
        optimizer.zero_grad()
        cond_probs = model.forward([H, dataset])
        loss = criterion(cond_probs, dataset)
        loss.backward()
        optimizer.step()

    # check if the final loss is smaller than the initial loss
    assert loss < first_loss, "Final loss is not smaller than initial loss"


if __name__ == "__main__":
    pytest.main([__file__])
