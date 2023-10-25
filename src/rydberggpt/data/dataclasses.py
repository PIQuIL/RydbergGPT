from abc import ABC
from dataclasses import dataclass
from typing import List

import torch
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch


@dataclass
class BaseGraph(ABC):
    """A base dataclass representing a graph configuration."""

    num_atoms: int
    graph_name: str
    Rb: float
    delta: float
    omega: float
    beta: float  # we cannot know the temperature of the system. Maybe remove?


@dataclass
class GridGraph(BaseGraph):
    n_rows: int
    n_cols: int


@dataclass
class Batch:
    graph: Data
    m_onehot: torch.Tensor
    # m_shifted_onehot: torch.Tensor


def custom_collate(batch: List[Batch]) -> Batch:
    """
    Custom collate function to handle Batch objects when creating a DataLoader.

    Args:
        batch (List[Batch]): A list of Batch objects to be collated.

    Returns:
        Batch: A single Batch object containing the collated data.
    """

    graph_batch = PyGBatch.from_data_list([b.graph for b in batch])

    # NOTE: The graphs, and measurement data are not of the same size. To ensure
    # a padded tensor suitable for the neural network, we use the to_dense_batch function. This ensures that our
    # data is padded with zeros.
    # see: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/to_dense_batch.html

    m_onehot = to_dense_batch(
        torch.cat([b.m_onehot for b in batch], axis=-2),
        batch=graph_batch.batch,
    )[0].to(torch.float32)

    return Batch(graph=graph_batch, m_onehot=m_onehot)
