from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset


@dataclass
class DatasetConfig:
    num_atoms: int
    num_samples: int
    delta: float


def load_dataset(delta_id: int) -> Tuple[np.ndarray, DatasetConfig]:
    """
    Loads a dataset from a numpy file and returns the dataset as a numpy array and a configuration object.

    Args:
        delta_id (int): An integer index specifying the delta value of the dataset to load.

    Returns:
        A tuple containing:
            - data (numpy.ndarray): A numpy array representing the loaded dataset. It is a two-dimensional array
              with the shape (num_samples, num_atoms), where num_samples is the number of samples in the dataset and
              num_atoms is the number of Rydberg atoms in each sample.
            - config (DatasetConfig): A dataclass object containing configuration parameters for the dataset.
    """
    list_delta = ["13.455", "4.955", "4.455", "-1.545"]
    delta = list_delta[delta_id]

    data = np.load(f"data/QMC/all_samples/delta_{delta}/uncorr_samples.npy")
    num_samples, num_atoms = data.shape

    config = DatasetConfig(
        num_atoms=num_atoms, num_samples=num_samples, delta=float(delta)
    )
    return data, config


def get_dataloaders(
    dataset: np.ndarray,
    config: DatasetConfig,
    train_ratio: float = 0.9,
    val_ratio: float = 0.1,
    test_ratio: float = 0.0,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns PyTorch DataLoaders for training, validation, and testing based on the given dataset,
    configuration, and splitting ratios.

    Args:
        dataset (numpy.ndarray): The input dataset as a numpy array of shape (num_samples, num_features).
        config (DatasetConfig): A dataclass containing configuration parameters for the dataset.
        train_ratio (float): The ratio of samples to use for training.
        val_ratio (float): The ratio of samples to use for validation.
        test_ratio (float): The ratio of samples to use for testing.
        batch_size (int): The batch size to use for training, validation, and testing.
        seed (int): The random seed to use for reproducibility.

    Returns:
        A tuple of PyTorch DataLoaders for training, validation, and testing.
    """
    assert (
        train_ratio + val_ratio + test_ratio == 1
    ), "train_ratio + val_ratio + test_ratio != 1"
    delta = config.delta

    # create a PyTorch TensorDataset from the numpy array, with the specified dtype
    delta_tensor = torch.full((len(dataset), config.num_atoms, 4), delta)
    dataset_tensor = torch.tensor(dataset, dtype=torch.int64)
    dataset = TensorDataset(dataset_tensor, delta_tensor)

    # calculate the size of the train, val, and test sets
    num_samples = len(dataset)
    num_train = int(num_samples * train_ratio)
    num_val = int(num_samples * val_ratio)
    num_test = num_samples - num_train - num_val

    # create random samplers for the train, val, and test sets
    indices = np.random.RandomState(seed=seed).permutation(num_samples)
    train_indices = indices[:num_train]
    val_indices = indices[num_train : num_train + num_val]
    test_indices = indices[num_train + num_val :]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # create DataLoaders for the train, val, and test sets
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        drop_last=True,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    batch_size = 32
    # load the dataset
    data, config = load_dataset(delta_id=0)

    # get the train/val/test dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        data, config, batch_size=batch_size
    )

    # iterate through the training data
    for batch in train_loader:
        inputs, condition = batch
        print(inputs.shape, condition.shape)
        assert inputs.shape == (batch_size, config.num_atoms)
        assert condition.shape == (batch_size, config.num_atoms, 4)
        assert inputs.dtype == torch.int64
        break

    print("Done!")
