import json
import logging
import os
import random
import uuid
from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd
import torch
import torch.distributed as dist
import yaml
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch as PyGBatch

from rydberggpt.data.dataclasses import Batch, custom_collate
from rydberggpt.data.loading.utils import contains_invalid_numbers
from rydberggpt.data.utils_graph import networkx_to_pyg_data
from rydberggpt.utils import time_and_log, to_one_hot


def get_rydberg_dataloader(
    batch_size: int = 10,
    test_size: float = 0.2,
    num_workers: int = 0,
    data_path: str = "dataset",
    buffer_size: int = 4,
) -> DataLoader:
    rank = dist.get_rank() if torch.distributed.is_initialized() else 0

    dataset_path = DatasetExtractor(data_path, {}).get_list_subset_ds_path()

    dataset = MainDataset(dataset_path, buffer_size)
    print(f"Length of dataset: {len(dataset)}")

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=custom_collate,
    )

    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=custom_collate,
    )
    return train_loader, val_loader


class DatasetExtractor(object):
    def __init__(self, base_dir: str, config: dict):
        """
        Initialize the DatasetExtractor.
        # TODO currently doesn't support scanning for specific features.

        Parameters:
            base_dir (str): The base directory containing the datasets.
            config_path (str): Path to the configuration file.
        """
        self.base_dir = base_dir
        self.config = config

    @time_and_log
    def get_list_subset_ds_path(self) -> None:
        """
        Read the folder structure of the base directory to identify paths to individual chunks

        """
        l_dirs = self.scan_directory()

        logging.info(f"Using the following folders: {l_dirs}")
        logging.info("Found %d folders containing datasets.", len(l_dirs))

        list_subset_ds_path = []
        for l_dir in l_dirs:
            subset_ds_dirs = self.scan_subset_ds_dirs(l_dir)
            logging.info(f"Found {len(subset_ds_dirs)} chunked datasets.")
            for subset_ds_dir in subset_ds_dirs:
                subset_ds_dir = os.path.join(self.base_dir, l_dir, subset_ds_dir)
                list_subset_ds_path.append(subset_ds_dir)
        logging.info(
            f"Found {len(list_subset_ds_path)} subset datasets matching the requirements."
        )
        return list_subset_ds_path

    def scan_directory(self) -> List[str]:
        """
        Scan the base directory for subdirectories and return a list of their names.

        Returns:
            List[str]: A list containing the names of all subdirectories in the base directory.
        """

        l_dirs = [
            d
            for d in os.listdir(self.base_dir)
            if os.path.isdir(os.path.join(self.base_dir, d))
        ]
        return l_dirs

    def scan_subset_ds_dirs(self, l_dir: str) -> List[str]:
        """
        Scan a given directory for subdirectories and return a list of their names.

        Args:
            l_dir (str): The directory to scan.

        Returns:
            List[str]: A list containing the names of all subdirectories in the given directory.
        """
        subset_dataset_dirs = [
            d
            for d in os.listdir(os.path.join(self.base_dir, l_dir))
            if os.path.isdir(os.path.join(self.base_dir, l_dir, d))
        ]
        return subset_dataset_dirs


class MainDataset(LightningDataModule):
    def __init__(self, list_subset_ds_path: List[str], num_subset_ds_in_mem: int = 1):
        self.list_subset_ds_path = list_subset_ds_path
        self.num_subset_ds_in_mem = num_subset_ds_in_mem

        # Subset dataset
        self.list_subset_ds = [SubsetDataset(path) for path in self.list_subset_ds_path]
        self.num_subset_ds = len(self.list_subset_ds)
        self.len_subset_ds = len(self.list_subset_ds[0])

        logging.info(
            f"Found {len(self.list_subset_ds_path)} subset datasets."
            f"\n Length for each subset dataset: {self.len_subset_ds}"
            f"\n Total length: {len(self)}"
        )

        # In memory dataset
        self.used_subset_ds_count = 0
        self.in_mem_ds = InMemoryDataset(self.num_subset_ds_in_mem, self.len_subset_ds)

        # shuffle the indices for randomness
        self.subset_ds_indices = list(range(len(self.list_subset_ds)))
        random.shuffle(self.subset_ds_indices)
        logging.info(f"Shuffled subset_ds indices: {self.subset_ds_indices}")

    def __len__(self) -> int:
        return len(self.list_subset_ds_path) * self.len_subset_ds

    def select_subset_ds_indices(self) -> List[int]:
        """
        Select random subset datasets to load into memory.

        Returns:
            List[int]: A list of indices of the subset datasets to load into memory.
        """
        start_idx = self.used_subset_ds_count
        end_idx = start_idx + self.num_subset_ds_in_mem
        selected_subset_ds_indices = self.subset_ds_indices[start_idx:end_idx]

        logging.info(
            f"Loading {selected_subset_ds_indices} random subset_ds into memory."
        )
        return selected_subset_ds_indices

    def select_and_load_subset_datasets(self):
        indices = self.select_subset_ds_indices()
        selected_subset_ds = [self.list_subset_ds[i] for i in indices]
        self.used_subset_ds_count += self.num_subset_ds_in_mem
        self.in_mem_ds.reload_buffer(selected_subset_ds)

    def reload_memory_if_needed(self):
        """
        Reload the in-memory dataset if needed.

        """
        if self.in_mem_ds.require_reload():
            self.in_mem_ds.free_memory()
            self.in_mem_ds.reset_samples_used()

            # check if full dataset can be loaded
            if self.num_subset_ds_in_mem > len(self):
                self.num_subset_ds_in_mem = len(self)
                logging.info(
                    f"Loading full dataset. num_subset_ds_in_mem set to {self.num_subset_ds_in_mem}"
                )

            self.select_and_load_subset_datasets()

            # reshuffle subset ds indices
            if self.used_subset_ds_count >= self.num_subset_ds:
                random.shuffle(self.subset_ds_indices)
                logging.info(
                    f"reshuffled subset paths indices: {self.subset_ds_indices}"
                )
                self.used_subset_ds_count = 0

    def __getitem__(self, idx) -> Batch:
        """
        Fetches a single sample from the dataset.

        Args:
            idx (int): The index of the sample to fetch.

        Returns:
            Batch: A dataclass containing the graph, the one-hot encoded input and target.
        """
        self.reload_memory_if_needed()
        return self.in_mem_ds[idx]


class SubsetDataset(object):
    instance_count = 0  # Class-level variable to keep count of instances

    def __init__(self, path):
        self._path: str = path
        self.index = SubsetDataset.instance_count
        SubsetDataset.instance_count += 1  # Increment the count for next instance

    @property
    def ds_path(self):
        return os.path.join(self._path, "dataset.h5")

    @property
    def graph_path(self):
        return os.path.join(self._path, "graph.json")

    @property
    def config_path(self):
        return os.path.join(self._path, "config.json")

    def load_ds(self):
        return pd.read_hdf(self.ds_path, key="data")

    @property
    def graph_data(self):
        with open(self.graph_path, "r") as f:
            graph_data = json.load(f)
        return graph_data

    @property
    def config(self):
        with open(self.config_path, "r") as f:
            config_data = json.load(f)
        return config_data

    @property
    def pyg_graph_data(self):
        """
        Convert a graph in node-link format to a PyG Data object.

        Args:
            graph_data (Dict): The graph in node-link format.
            config_data (Dict): The configuration data for the graph.

        Returns:
            PyG Data: The graph as a PyG Data object.

        """
        node_features = torch.tensor(
            [
                self.config["delta"],
                self.config["omega"],
                self.config["beta"],
                self.config["Rb"],
            ],
            dtype=torch.float32,
        )
        graph_nx = nx.node_link_graph(self.graph_data)
        pyg_graph = networkx_to_pyg_data(graph_nx, node_features)
        return pyg_graph

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        df = pd.read_hdf(self.ds_path, key="data")
        self.len_sub_ds = df.shape[0]
        del df
        return self.len_sub_ds

    @classmethod
    def index(cls):
        return cls.instance_count


@dataclass
class InMemoryDataclass:
    df: List[pd.DataFrame]
    pyg_graph: List[PyGBatch]


class InMemoryDataset(object):
    def __init__(self, num_subset_ds_in_mem: int, len_subset_ds: int):
        self.num_subset_ds_in_mem = num_subset_ds_in_mem
        self.len_subset_ds = len_subset_ds
        self.in_memory_dataset = []
        self.indices = []
        self.samples_used = 0  # Counter for samples used

    def require_reload(self) -> bool:
        """
        Check if the in-memory dataset needs to be reloaded.

        Returns:
            bool: True if the in-memory dataset needs to be reloaded, False otherwise.

        """
        if not self.in_memory_dataset or self.samples_used == len(self.indices):
            return True
        else:
            return False

    def reset_samples_used(self):
        """
        Reset the counter for samples used.
        """
        self.samples_used = 0

    @time_and_log
    def reload_buffer(self, list_subset_ds: List[SubsetDataset]):
        """
        Reload the in-memory dataset with the given subset datasets.

        Args:
            list_subset_ds (List[SubsetDataset]): A list of subset datasets to load into memory.

        """
        for i, subset_ds in enumerate(list_subset_ds):
            df = subset_ds.load_ds()
            self.in_memory_dataset.append(
                InMemoryDataclass(df, subset_ds.pyg_graph_data)
            )
            self.indices.extend([(i, j) for j in range(len(df))])
        # Shuffle the indices for randomness
        random.shuffle(self.indices)

        assert len(self.in_memory_dataset) == len(
            list_subset_ds
        ), "The number of graphs loaded is not equal to the number of subset datasets loaded."

    def __len__(self) -> int:
        return self.num_subset_ds_in_mem * self.len_subset_ds

    def free_memory(self):
        """
        Free memory by clearing the current buffers.

        """
        logging.info(f"Freeing memory.")
        self.in_memory_dataset.clear()

    def get_sample(self, df, idx):
        return torch.tensor(df.iloc[idx]["measurement"], dtype=torch.bool)

    def __getitem__(self, idx: int) -> Batch:
        """
        Fetches a single sample from the dataset.

        Args:
            idx (int): The index of the sample to fetch.

        Returns:
            Batch: A dataclass containing the graph, the one-hot encoded input and target.

        """

        # Determine the chunk and sample index
        # NOTE: This is necessary because the in memory dataset is smaller than the full dataset
        idx_new = idx % len(self)
        subset_ds_idx, sample_idx = self.indices[idx_new]

        data = self.in_memory_dataset[subset_ds_idx]
        measurement = self.get_sample(data.df, sample_idx)

        # Prepare one-hot encoded input and target
        m_onehot = to_one_hot(measurement, 2)
        _, dim = m_onehot.shape
        # m_shifted_onehot = torch.cat((torch.zeros(1, dim), m_onehot[:-1]), dim=0)

        # Check for NaN/Inf values
        # if contains_invalid_numbers(m_onehot) or contains_invalid_numbers(
        #     m_shifted_onehot
        # ):
        #     dataset_path = self.chunk_paths[subset_ds_idx]
        #     logging.info(
        #         f"Invalid numbers found in dataset: {dataset_path}, sample: {sample_idx}"
        #     )

        self.samples_used += 1

        # Return the sample as a Batch object
        return Batch(
            graph=data.pyg_graph,
            m_onehot=m_onehot,
            # m_shifted_onehot=m_shifted_onehot,
        )
