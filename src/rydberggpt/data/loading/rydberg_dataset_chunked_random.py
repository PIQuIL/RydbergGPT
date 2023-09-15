import json
import logging
import random
import uuid
from typing import List

import pandas as pd
import torch
from torch.utils.data import DataLoader

from rydberggpt.data.dataclasses import Batch, custom_collate
from rydberggpt.data.loading.base_dataset import BaseDataset
from rydberggpt.data.loading.utils import contains_invalid_numbers
from rydberggpt.utils import to_one_hot, track_memory_usage


def get_chunked_random_dataloader(
    batch_size: int = 10,
    test_size: float = 0.2,
    num_workers: int = 0,
    data_path: str = "dataset",
    chunks_in_memory: int = 4,
) -> DataLoader:
    dataset = ChunkedDatasetRandom(data_path, chunks_in_memory)
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


class ChunkedDatasetRandom(BaseDataset):
    def __init__(self, base_dir: str, num_chunks_in_memory: int = 50):
        super().__init__(base_dir)
        self.num_chunks_in_memory = num_chunks_in_memory
        self.current_dfs = []
        self.current_configs = []
        self.current_graphs = []
        self.current_py_graphs = []
        self.current_indices = []
        self.samples_used = 0  # Counter for samples used
        self._load_sub_dataset()

    def _reload_buffer(self, selected_chunk_indices: List[int]):
        """
        Load data from the selected chunks into memory.

        Args:
            selected_chunk_indices (List[int]): A list of indices of the chunks to load.

        """
        for i, chunk_idx in enumerate(selected_chunk_indices):
            df = pd.read_hdf(self.chunk_paths[chunk_idx], key="data")
            self.current_dfs.append(df)

            with open(self.config_paths[chunk_idx], "r") as f:
                config_data = json.load(f)
                self.current_configs.append(config_data)

            with open(self.graph_paths[chunk_idx], "r") as f:
                graph_data = json.load(f)
                self.current_graphs.append(graph_data)

            self.current_py_graphs.append(self._get_pyg_graph(graph_data, config_data))
            self.current_indices.extend([(i, j) for j in range(len(df))])

        assert len(self.current_py_graphs) == len(
            selected_chunk_indices
        ), "The number of graphs loaded is not equal to the number of chunks loaded."
        assert len(self.current_configs) == len(
            selected_chunk_indices
        ), "The number of configs loaded is not equal to the number of chunks loaded."
        assert len(self.current_graphs) == len(
            selected_chunk_indices
        ), "The number of graphs loaded is not equal to the number of chunks loaded."

        # Shuffle the indices for randomness
        random.shuffle(self.current_indices)

    def _load_sub_dataset(self, seed: int = None):
        if self.num_chunks_in_memory > len(self.chunk_paths):
            self.num_chunks_in_memory = len(self.chunk_paths)
            print(
                f"Loading full dataset. num_chunks_in_memory set to {self.num_chunks_in_memory}"
            )

        # Determine the start and end indices for the chunks to load based on the counter
        start_idx = self.current_chunk_counter
        end_idx = start_idx + self.num_chunks_in_memory

        # Select chunk indices based on the shuffled chunk_indices and num_chunks_in_memory
        selected_chunk_indices = self.chunk_indices[start_idx:end_idx]
        logging.info(f"Loading {selected_chunk_indices} random chunks into memory.")

        self._reload_buffer(selected_chunk_indices)

        # Update the current_chunk_counter
        self.current_chunk_counter += self.num_chunks_in_memory

        # Reset counter if we have loaded all chunks (optional, based on your needs)
        if self.current_chunk_counter >= len(self.chunk_paths):
            self.current_chunk_counter = 0
            random.shuffle(
                self.chunk_indices
            )  # Reshuffle the chunk indices for the next iteration

    def __getitem__(self, idx: int) -> Batch:
        chunk_idx, sample_idx = self.current_indices[idx]

        # Fetch data sample from the current chunk
        df = self.current_dfs[chunk_idx]
        pyg_graph = self.current_py_graphs[chunk_idx]

        measurement = torch.tensor(df.iloc[sample_idx]["measurement"], dtype=torch.bool)

        # Prepare one-hot encoded input and target
        m_onehot = to_one_hot(measurement, 2)
        _, dim = m_onehot.shape
        m_shifted_onehot = torch.cat((torch.zeros(1, dim), m_onehot[:-1]), dim=0)

        # Check for NaN/Inf values
        if contains_invalid_numbers(m_onehot) or contains_invalid_numbers(
            m_shifted_onehot
        ):
            dataset_path = self.chunk_paths[chunk_idx]
            logging.info(
                f"Invalid numbers found in dataset: {dataset_path}, sample: {sample_idx}"
            )

        # Check if all samples have been used once
        self.samples_used += 1
        if self.samples_used >= len(self.current_indices):
            logging.info("Loading new chunks.")
            self.samples_used = 0  # Reset counter
            # Clear the current data
            self.current_dfs.clear()
            self.current_configs.clear()
            self.current_graphs.clear()
            self.current_indices.clear()
            self.current_py_graphs.clear()
            self._load_sub_dataset()  # Load new random chunks

        return Batch(
            graph=pyg_graph,
            m_onehot=m_onehot,
            m_shifted_onehot=m_shifted_onehot,
        )

    def __len__(self):
        return len(self.current_indices)
