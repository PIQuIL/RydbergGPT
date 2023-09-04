import os
from typing import Dict, Tuple

import networkx as nx
import pandas as pd
import torch
from torch.utils.data import Dataset

from rydberggpt.data.utils_graph import networkx_to_pyg_data


class BaseDataset(Dataset):
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

    def __getitem__(self, idx):
        raise NotImplementedError

    def _get_pyg_graph(self, graph_data: Dict, config_data: Dict):
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
