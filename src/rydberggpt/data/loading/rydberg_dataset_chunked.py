import json
import os
from typing import Dict, Tuple

import networkx as nx
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.data import Data

from rydberggpt.data.dataclasses import Batch, custom_collate
from rydberggpt.data.loading.base_dataset import BaseDataset
from rydberggpt.data.utils_graph import networkx_to_pyg_data
from rydberggpt.utils import to_one_hot


# NOTE not in use
def get_chunked_dataloader(
    batch_size: int = 32,
    test_size: float = 0.2,
    num_workers: int = 0,
    data_path: str = "dataset",
) -> Tuple[DataLoader, DataLoader]:
    # Initialize the dataset
    full_dataset = ChunkedDatasetPandasRandomAccess(data_path)
    assert (
        len(full_dataset) > 0
    ), "Dataset is empty. Check that the data_path is correct."

    # Compute the lengths of training and validation datasets
    # total_samples = len(full_dataset)
    # val_samples = int(test_size * total_samples)
    # train_samples = total_samples - val_samples

    # Split the dataset into training and validation subsets
    # train_dataset, val_dataset = random_split(
    # full_dataset, [train_samples, val_samples]
    # )

    # Create dataloaders
    train_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate,
    )
    val_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate,
    )
    return train_loader, val_loader


class ChunkedDatasetPandasRandomAccess(BaseDataset):  # Inherit from BaseDataset
    def __init__(self, base_dir: str):
        super().__init__(base_dir)  # Call the constructor of the base class

    def __getitem__(self, idx: int) -> Batch:
        """
        Fetch a data sample given its index.

        Args:
            idx (int): The index of the data sample to fetch.

        Returns:
            Batch: A batch containing the graph data, one-hot encoded measurement data,
                   and one-hot encoded shifted measurement data.
        """
        chunk_idx = 0
        while idx >= self.lengths[chunk_idx]:
            idx -= self.lengths[chunk_idx]
            chunk_idx += 1

        measurement, config_data, graph_data = self._load_data_sample(chunk_idx, idx)

        pyg_graph = self._get_pyg_graph(graph_data, config_data)

        # prepare one hot encoded input and target
        m_onehot = to_one_hot(measurement, 2)  # because Rydberg states are 0 or 1

        _, dim = m_onehot.shape
        m_shifted_onehot = torch.cat((torch.zeros(1, dim), m_onehot[:-1]), dim=0)
        return Batch(
            graph=pyg_graph,
            m_onehot=m_onehot,
            m_shifted_onehot=m_shifted_onehot,
        )

    def _load_data_sample(
        self, chunk_idx: int, idx: int
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        """
        Load a single data sample from the chunked dataset.

        Args:
            chunk_idx (int): Index of the chunk to load the data from.
            idx (int): Index of the data sample within the chunk.

        Returns:
            measurement (torch.Tensor): Measurement data as a 1D tensor of integers.
            config_data (Dict): Configuration data loaded from the config.json file.
            graph_data (Dict): Graph data loaded from the graph.json file.
        """
        # Load measurement data from the .h5 file
        df = pd.read_hdf(
            self.chunk_paths[chunk_idx], key="data", start=idx, stop=idx + 1
        )
        measurement = torch.tensor(df["measurement"].iloc[0], dtype=torch.int64)

        # Load configuration data from the config.json file
        with open(self.config_paths[chunk_idx], "r") as config_file:
            config_data = json.load(config_file)

        # Load graph data from the graph.json file
        with open(self.graph_paths[chunk_idx], "r") as graph_file:
            graph_data = json.load(graph_file)

        return measurement, config_data, graph_data
