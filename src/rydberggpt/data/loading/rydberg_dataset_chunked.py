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
from rydberggpt.data.utils_graph import networkx_to_pyg_data
from rydberggpt.utils import to_one_hot


def get_chunked_dataloader(
    batch_size: int = 32,
    test_size: float = 0.2,
    num_workers: int = 0,
    data_path: str = "dataset",
) -> Tuple[DataLoader, DataLoader]:
    # Initialize the dataset
    full_dataset = ChunkedDatasetPandasRandomAccess(data_path)

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


class ChunkedDatasetPandasRandomAccess(Dataset):
    """
    A dataset class that reads data from chunked datasets stored in an HDF5 format.
    Each chunk is a combination of measurement data, graph data, and configuration data.
    """

    def __init__(self, base_dir: str):
        """
        Initialize the dataset with the base directory containing the chunked datasets.

        Args:
            base_dir (str): The directory containing the chunked datasets.
        """
        self.base_dir = base_dir
        self.chunk_paths = []
        self.graph_paths = []
        self.config_paths = []
        self.lengths = []
        self.total_length = 0
        self._read_folder_structure()

    def _read_folder_structure(self) -> None:
        """
        Read the folder structure of the base directory to identify paths to individual chunks,
        their associated graph and configuration data.
        """
        # List all directories with chunked datasets
        l_dirs = [
            d
            for d in os.listdir(self.base_dir)
            if os.path.isdir(os.path.join(self.base_dir, d))
        ]

        for l_dir in l_dirs:
            chunked_dataset_dirs = [
                d
                for d in os.listdir(os.path.join(self.base_dir, l_dir))
                if os.path.isdir(os.path.join(self.base_dir, l_dir, d))
            ]
            for chunked_dataset_dir in chunked_dataset_dirs:
                chunk_dir = os.path.join(self.base_dir, l_dir, chunked_dataset_dir)
                df_shape = pd.read_hdf(
                    os.path.join(chunk_dir, "dataset.h5"), key="data"
                ).shape
                self.chunk_paths.append(os.path.join(chunk_dir, "dataset.h5"))
                self.graph_paths.append(os.path.join(chunk_dir, "graph.json"))
                self.config_paths.append(os.path.join(chunk_dir, "config.json"))
                self.lengths.append(df_shape[0])
                self.total_length += df_shape[0]

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return self.total_length

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

    def _get_pyg_graph(self, graph_data: Dict, config_data: Dict) -> Data:
        """
        Convert the graph data to a PyG Data object.

        Args:
            graph_data (Dict): Graph data loaded from the graph.json file.
            config_data (Dict): Configuration data loaded from the config.json file.

        Returns:
            Data: A PyG Data object representing the graph.
        """

        node_features = torch.tensor(
            [
                config_data["delta"],
                config_data["omega"],
                config_data["beta"],
                config_data["Rb"],
            ],
            dtype=torch.float32,
        )
        graph_nx = nx.node_link_graph(graph_data)
        pyg_graph = networkx_to_pyg_data(graph_nx, node_features)
        return pyg_graph
