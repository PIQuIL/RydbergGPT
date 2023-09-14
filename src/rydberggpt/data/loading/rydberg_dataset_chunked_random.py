import json
import logging
import random
import uuid

import pandas as pd
import torch
from torch.utils.data import DataLoader

from rydberggpt.data.dataclasses import Batch, custom_collate
from rydberggpt.data.loading.base_dataset import BaseDataset
from rydberggpt.data.loading.utils import contains_invalid_numbers
from rydberggpt.utils import to_one_hot, track_memory_usage

logger = logging.getLogger(__name__)


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
    def __init__(self, base_dir: str, num_chunks_in_memory: int = 300):
        super().__init__(base_dir)
        self.num_chunks_in_memory = num_chunks_in_memory
        self.current_dfs = []
        self.current_configs = []
        self.current_graphs = []
        self.current_indices = []
        self.samples_used = 0  # Counter for samples used
        self._load_random_chunks()

    # TODO is it properly randomized for multi GPU training?
    # TODO dirty fix with uuid for now
    @track_memory_usage
    def _load_random_chunks(self, seed=None):
        random.seed(uuid.uuid4().int & (1 << 32) - 1)

        if self.num_chunks_in_memory > len(self.chunk_paths):
            self.num_chunks_in_memory = len(self.chunk_paths)
            print(
                f"Loading full dataset. num_chunks_in_memory set to {self.num_chunks_in_memory}"
            )

        # Randomly choose chunk indices
        random_chunk_indices = random.sample(
            range(len(self.chunk_paths)), self.num_chunks_in_memory
        )
        print(f"Random chunk indices: {random_chunk_indices}")

        # Clear the current data
        self.current_dfs.clear()
        self.current_configs.clear()
        self.current_graphs.clear()
        self.current_indices.clear()

        # Load data for the selected chunks
        for i, chunk_idx in enumerate(random_chunk_indices):
            df = pd.read_hdf(self.chunk_paths[chunk_idx], key="data")
            self.current_dfs.append(df)

            with open(self.config_paths[chunk_idx], "r") as f:
                self.current_configs.append(json.load(f))

            with open(self.graph_paths[chunk_idx], "r") as f:
                self.current_graphs.append(json.load(f))

            self.current_indices.extend([(i, j) for j in range(len(df))])

        # Shuffle the indices for randomness
        random.shuffle(self.current_indices)

    def __getitem__(self, idx: int) -> Batch:
        chunk_idx, sample_idx = self.current_indices[idx]

        # Fetch data sample from the current chunk
        df = self.current_dfs[chunk_idx]
        config_data = self.current_configs[chunk_idx]
        graph_data = self.current_graphs[chunk_idx]

        measurement = torch.tensor(df.iloc[sample_idx]["measurement"], dtype=torch.bool)
        pyg_graph = self._get_pyg_graph(graph_data, config_data)

        # Prepare one-hot encoded input and target
        m_onehot = to_one_hot(measurement, 2)
        _, dim = m_onehot.shape
        m_shifted_onehot = torch.cat((torch.zeros(1, dim), m_onehot[:-1]), dim=0)

        # Check for NaN/Inf values
        if contains_invalid_numbers(m_onehot) or contains_invalid_numbers(
            m_shifted_onehot
        ):
            dataset_path = self.chunk_paths[chunk_idx]
            logger.info(
                f"Invalid numbers found in dataset: {dataset_path}, sample: {sample_idx}"
            )
            print(
                f"Invalid numbers found in dataset: {dataset_path}, sample: {sample_idx}"
            )

        # Check if all samples have been used once
        self.samples_used += 1
        if self.samples_used >= len(self.current_indices):
            logger.info("All samples used once. Loading new random chunks.")
            self.samples_used = 0  # Reset counter
            self._load_random_chunks()  # Load new random chunks

        return Batch(
            graph=pyg_graph,
            m_onehot=m_onehot,
            m_shifted_onehot=m_shifted_onehot,
        )

    def __len__(self):
        return len(self.current_indices)
