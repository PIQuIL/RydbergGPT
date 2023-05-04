from dataclasses import dataclass
from typing import List

import torch
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.data import Data


@dataclass
class Batch:
    graph: Data
    delta: torch.float
    omega: torch.float
    beta: torch.float
    m_onehot: torch.Tensor
    m_shifted_onehot: torch.Tensor


def custom_collate(batch: List[Batch]) -> Batch:
    """
    Custom collate function to handle Batch objects when creating a DataLoader.

    Args:
        batch (List[Batch]): A list of Batch objects to be collated.

    Returns:
        Batch: A single Batch object containing the collated data.
    """
    batch = Batch(
        graph=PyGBatch.from_data_list([b.graph for b in batch]),
        delta=torch.tensor([b.delta for b in batch], dtype=torch.float32),
        omega=torch.tensor([b.omega for b in batch], dtype=torch.float32),
        beta=torch.tensor([b.beta for b in batch], dtype=torch.float32),
        m_onehot=torch.stack([b.m_onehot for b in batch]).to(torch.float32),
        m_shifted_onehot=torch.stack([b.m_shifted_onehot for b in batch]).to(
            torch.float32
        ),
    )
    return batch
