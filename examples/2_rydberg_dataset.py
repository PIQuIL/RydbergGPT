from dataclasses import dataclass

import matplotlib.pyplot as plt

# import uncorr_samples.npy file located in data/QMC/all_samples/delta_13.455/uncorr_samples.npy
# load data
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset

list_delta = ["13.455", "4.955", "4.455", "-1.545"]
delta = list_delta[0]

data = np.load(f"data/QMC/all_samples/delta_{delta}/uncorr_samples.npy")
num_samples, num_atoms = data.shape
# print(data.shape, data)


# create a dataclass to store the configuration parameters
@dataclass
class DatasetConfig:
    num_atoms: int
    delta: float


dataset_config = DatasetConfig(num_atoms=num_atoms, delta=float(delta))


# set the random seed for reproducibility
np.random.seed(42)

# load the dataset as a numpy array
# create a PyTorch TensorDataset from the numpy array
dataset = TensorDataset(torch.from_numpy(data))

# define the train/val/test split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# calculate the size of the train, val, and test sets
num_samples = len(dataset)
num_train = int(num_samples * train_ratio)
num_val = int(num_samples * val_ratio)
num_test = num_samples - num_train - num_val

# create random samplers for the train, val, and test sets
indices = np.random.permutation(num_samples)
train_indices = indices[:num_train]
val_indices = indices[num_train : num_train + num_val]
test_indices = indices[num_train + num_val :]
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

# create DataLoaders for the train, val, and test sets
batch_size = 32  # adjust as needed
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

# iterate through the training data
for batch in train_loader:
    # extract the inputs from the batch
    inputs = batch[0]
    print(inputs)
    break
