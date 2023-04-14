import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

# def one_hot(data, num_classes):
#     """
#     Convert a one-dimensional array of labels into a matrix of one-hot labels.
#     :param data: numpy array of shape (num_samples,)
#     :param num_classes: number of classes
#     :return: numpy array of shape (num_samples, num_classes)
#     """
#     return np.eye(num_classes)[data]


# def measurement2readout_obs(measurments):
#     if isinstance(measurments, torch.Tensor):
#         measurments = torch2numpy(measurments)
#     assert measurments.min() >= 0 and measurments.max() <= 5
#     observables = measurments // 2
#     readouts = 2.0 * (measurments % 2) - 1
#     return readouts, observables


# def torch2numpy(*tensors):
#     arrays = []
#     for tensor in tensors:
#         arrays.append(tensor.detach().cpu().numpy())
#     if len(arrays) == 1:
#         return arrays[0]
#     else:
#         return tuple(arrays)


# def numpy2torch(*arrays, device=torch.device("cpu")):
#     tensors = []
#     for array in arrays:
#         tensors.append(torch.from_numpy(array).to(device))
#     if len(tensors) == 1:
#         return tensors[0]
#     else:
#         return tuple(tensors)


def get_dummy_dataset(n_atoms: int, batch_size: int, dim: int) -> torch.Tensor:
    """Generates a random dataset for testing purposes.

    The dataset has shape [batch_size, n_atoms, dim], where each dim contains only a single 1 and
    the other elements are set to 0. The dataset is then converted to a PyTorch tensor of type
    torch.float32.

    Args:
        n_atoms (int): The number of atoms in the dataset.
        batch_size (int): The number of examples in the dataset.
        dim (int): The number of dimensions in each example.

    Returns:
        torch.Tensor: A PyTorch tensor containing the random dataset, with shape [batch_size,
            n_atoms, dim].
    """
    # Generate random dataset with shape [batch_size, n_atoms, dim] where each dim contains only
    # a single 1
    dataset = np.zeros((batch_size, n_atoms, dim))
    for b in range(batch_size):
        for t in range(n_atoms):
            dataset[b, t, np.random.randint(dim)] = 1

    # Check dataset shape
    assert dataset.shape == (batch_size, n_atoms, dim), "dataset has wrong shape"

    # Check that each dim contains only a single 1
    assert np.allclose(
        np.sum(dataset, axis=2), np.ones((batch_size, n_atoms))
    ), "dataset contains more than one 1 per dim"

    # Check that dataset contains only 0 and 1
    assert np.allclose(
        dataset, dataset.astype(bool)
    ), "dataset contains values other than 0 and 1"

    # Convert dataset to PyTorch tensor of type torch.float32
    dataset = torch.from_numpy(dataset).float()

    return dataset
