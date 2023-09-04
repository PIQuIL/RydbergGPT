# First, let's import the necessary modules and helper functions that we're going to use

import json
import os
from typing import Dict, List, Tuple

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
def get_streaming_dataloader(
    batch_size: int = 32,
    test_size: float = 0.2,
    num_workers: int = 0,
    data_path: str = "dataset",
) -> Tuple[DataLoader, DataLoader]:
    # Initialize the dataset
    full_dataset = StreamingDataLoader(data_path)
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


class StreamingDataLoader(BaseDataset):
    def __init__(self, base_dir: str):
        super().__init__(base_dir)
        # Initialize to -1 to indicate no chunk is currently loaded
        self.current_chunk_idx = -1
        self.current_df = None
        self.current_graph_data = None
        self.current_config_data = None
        self.current_sample_idx = 0  # Initialize to 0

    def _load_next_chunk(self) -> None:
        self.current_chunk_idx = (self.current_chunk_idx + 1) % len(
            self.chunk_paths
        )  # Loop back to the first chunk after the last
        self.current_sample_idx = 0  # Reset the sample index
        self.current_df = pd.read_hdf(
            self.chunk_paths[self.current_chunk_idx], key="data"
        )

        # Load the graph_data and config_data for the current chunk
        with open(self.graph_paths[self.current_chunk_idx], "r") as graph_file:
            self.current_graph_data = json.load(graph_file)
        with open(self.config_paths[self.current_chunk_idx], "r") as config_file:
            self.current_config_data = json.load(config_file)

    def __getitem__(self, idx: int) -> Batch:
        if self.current_df is None or self.current_sample_idx >= len(self.current_df):
            self._load_next_chunk()

        measurement, config_data, graph_data = self._load_data_sample()

        pyg_graph = self._get_pyg_graph(graph_data, config_data)

        m_onehot = to_one_hot(measurement, 2)  # because Rydberg states are 0 or 1
        _, dim = m_onehot.shape
        m_shifted_onehot = torch.cat((torch.zeros(1, dim), m_onehot[:-1]), dim=0)

        self.current_sample_idx += 1  # Increment the sample index for the current chunk
        return Batch(
            graph=pyg_graph,
            m_onehot=m_onehot,
            m_shifted_onehot=m_shifted_onehot,
        )

    def _load_data_sample(self) -> Tuple[torch.Tensor, Dict, Dict]:
        df_row = self.current_df.iloc[self.current_sample_idx]
        measurement = torch.tensor(df_row["measurement"], dtype=torch.int64)
        return measurement, self.current_config_data, self.current_graph_data
