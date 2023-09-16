import logging
import os
import random
import uuid
from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd
import torch
from torch.utils.data import Dataset

from rydberggpt.data.utils_graph import networkx_to_pyg_data
from rydberggpt.utils import track_memory_usage


class BaseDataset(Dataset):
    def __init__(self, base_dir: str, rank: int = 0):
        """
        Initialize the dataset with the base directory containing the chunked datasets.

        Args:
            base_dir (str): The directory containing the chunked datasets.
        """
        self.base_dir = base_dir
        self.rank = rank
        random.seed(self.rank)

        self.chunk_paths = []
        self.graph_paths = []
        self.config_paths = []
        self.lengths = []
        self.total_length = 0
        self.len_sub_dataset = None

        self._read_folder_structure()
        self.current_chunk_counter = 0
        self.chunk_indices = list(range(len(self.chunk_paths)))
        random.shuffle(self.chunk_indices)

        # translate indices to the chunk path for storing in the log
        self.shuffled_chunk_path = [self.chunk_paths[i] for i in self.chunk_indices]
        self.current_chunk_counter = 0  # Initialize the chunk counter
        logging.info(
            f"GPU {self.rank}: Shuffled chunk paths indices: {self.chunk_indices}"
        )
        # logging.info(
        #     f"GPU {self.rank}: Shuffled chunk paths: {self.shuffled_chunk_path}"
        # )

    def _scan_directories(self) -> List[str]:
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

    def _scan_chunked_dataset_dirs(self, l_dir: str) -> List[str]:
        """
        Scan a given directory for subdirectories and return a list of their names.

        Args:
            l_dir (str): The directory to scan.

        Returns:
            List[str]: A list containing the names of all subdirectories in the given directory.
        """
        chunked_dataset_dirs = [
            d
            for d in os.listdir(os.path.join(self.base_dir, l_dir))
            if os.path.isdir(os.path.join(self.base_dir, l_dir, d))
        ]
        return chunked_dataset_dirs

    def _get_len_sub_dataset(self, chunk_dir: str) -> None:
        """
        Estimate the length (number of rows) of a given chunked dataset.
        This function sets the `len_sub_dataset` attribute of the class
        based on the shape of the first encountered chunked dataset.
        NOTE: It assumes that every chunked dataset has the same size!

        Args:
            chunk_dir (str): Directory path containing the chunked dataset.
        """
        df = pd.read_hdf(os.path.join(chunk_dir, "dataset.h5"), key="data")
        self.len_sub_dataset = df.shape[0]
        del df

    def _append_files_from_chunked_dir(self, chunk_dir: str) -> None:
        """
        Append paths of the chunked dataset, graph, and config files
        from the given directory to the respective class attributes.
        Also updates the total length and lengths attributes.

        Args:
            chunk_dir (str): Directory path containing the chunked dataset.
        """
        self.chunk_paths.append(os.path.join(chunk_dir, "dataset.h5"))
        self.graph_paths.append(os.path.join(chunk_dir, "graph.json"))
        self.config_paths.append(os.path.join(chunk_dir, "config.json"))
        self.lengths.append(self.len_sub_dataset)
        self.total_length += self.len_sub_dataset

    @track_memory_usage
    def _read_folder_structure(self) -> None:
        """
        Read the folder structure of the base directory to identify paths to individual chunks,
        their associated graph and configuration data.
        """
        l_dirs = self._scan_directories()
        logging.info(f"Using the following folders: {l_dirs}")
        logging.info("Found %d folders containing datasets.", len(l_dirs))
        for l_dir in l_dirs:
            chunked_dataset_dirs = self._scan_chunked_dataset_dirs(l_dir)
            logging.info(f"Found {len(chunked_dataset_dirs)} chunked datasets.")
            for chunked_dataset_dir in chunked_dataset_dirs:
                chunk_dir = os.path.join(self.base_dir, l_dir, chunked_dataset_dir)
                # NOTE scanning every dataset for its size is slow thus we just take the first one
                if self.len_sub_dataset is None:
                    self._get_len_sub_dataset(chunk_dir)
                    logging.info(
                        f"Estimated length of each sub dataset: {self.len_sub_dataset}"
                    )
                self._append_files_from_chunked_dir(chunk_dir)
        logging.info(
            f"Found {len(self.chunk_paths)} chunks. Total length: {self.total_length}"
        )

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return self.total_length

    def __getitem__(self, idx):
        raise NotImplementedError

    # NOTE this function is really slow. Run it once and save the results.
    # @track_memory_usage
    def _get_pyg_graph(self, graph_data: Dict, config_data: Dict):
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
