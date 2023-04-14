import numpy as np
import torch
import torch.nn as nn

# First we generate a random dataset with the shape [B, T, Q], where B is the batch size, T the index of the rydberg atom and Q (is always 2) the dimension of the state vector which can be either 0 or 1.

# generate dataset with B=1
B = 1  # batch size
T = 3  # number of rydberg atoms
Q = 2  # always 2, since Rydberg atoms are ideally a 2-level system.

# seed
np.random.seed(0)

# generate random dataset with shape [B, T, Q] where each Q dim contains only a single 1
dataset = np.zeros((B, T, Q))
for b in range(B):
    for t in range(T):
        dataset[b, t, np.random.randint(Q)] = 1

# check dataset shape
assert dataset.shape == (B, T, Q), "dataset has wrong shape"
print(dataset)

# check that each Q dim contains only a single 1
assert np.allclose(
    np.sum(dataset, axis=2), np.ones((B, T))
), "dataset contains more than one 1 per Q dim"


# check that dataset contains only 0 and 1
assert np.allclose(
    dataset, dataset.astype(bool)
), "dataset contains values other than 0 and 1"

# The transformer gives us an output of the shape [B, T, 2], where B is the batch size, T the index of the rydberg atom and 2 the dimension of the state vector which now represents the probability of being either in state 0 or state 1.

# generate a possible output from the transformer
output = np.random.rand(B, T, 2)
# apply softmax to get probabilities
output = np.exp(output) / np.sum(np.exp(output), axis=2, keepdims=True)

# check that probabilities sum up to 1
assert np.allclose(np.sum(output, axis=2), 1.0), "probabilities do not sum up to 1"

# convert dataset and output to pytorch tensors
dataset = torch.from_numpy(dataset).float()
output = torch.from_numpy(output).float()

# Next let us define the loss function.


def loss_fn(dataset: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    """Calculates the element-wise product of two input tensors and returns the sum of the logarithm
    of the resulting tensor along the batch dimension.

    Args:
        dataset (torch.Tensor): The input tensor containing the dataset features. It has shape
            `[B, T, Q]`.
        output (torch.Tensor): The input tensor containing the model predictions. It has shape
            `[B, T, Q]`.

    Returns:
        torch.Tensor: A scalar tensor containing the loss value.
    """
    # elementwise multiplication, resulting tensor has shape [B, T, Q]
    product = dataset * output
    assert product.shape == (B, T, Q), "product has wrong shape"
    # sum along feature dimension, resulting tensor has shape [B, T]
    product = torch.sum(product, dim=2)
    assert product.shape == (B, T), "product has wrong shape"
    # product along sequence dimension, resulting tensor has shape [B]
    product = torch.prod(product, dim=1)
    assert product.shape == (B,), "product has wrong shape"
    # sum of logarithm of product, resulting tensor has shape []
    loss = -torch.sum(torch.log(product))
    return loss


loss = loss_fn(output, dataset)
print(loss)
# loss = loss_fn_2(output, dataset)
# print(loss)
# loss = loss_fn_3(output, dataset)
# print(loss)

# next lets sample from the output tensor


# create a transformer nn using pytorch
# https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
