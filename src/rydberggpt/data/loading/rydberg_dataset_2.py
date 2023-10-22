import logging
import random
from typing import Tuple

import networkx as nx
import pandas as pd
import torch
from torchdata.dataloader2 import (
    DataLoader2,
    DistributedReadingService,
    InProcessReadingService,
    MultiProcessingReadingService,
    PrototypeMultiProcessingReadingService,
)
from torchdata.datapipes.iter import FileLister, IterDataPipe

from rydberggpt.data.dataclasses import Batch, custom_collate
from rydberggpt.data.utils_graph import pyg_graph_data
from rydberggpt.utils import to_one_hot

logging.basicConfig(level=logging.INFO)


def get_rydberg_dataloader_2(
    batch_size: int = 10,
    test_size: float = 0.2,
    num_workers: int = 0,
    data_path: str = "dataset",
    buffer_size: int = 4,
) -> Tuple[DataLoader2, DataLoader2]:
    datapipe = build_datapipes(
        root_dir=data_path, batch_size=batch_size, buffer_size=buffer_size
    )

    # rs = DistributedReadingService()
    # rs = MultiProcessingReadingService(num_workers=num_workers)
    # rs = PrototypeMultiProcessingReadingService(num_workers=num_workers)
    rs = InProcessReadingService()

    train_loader = DataLoader2(datapipe=datapipe, reading_service=rs)

    return train_loader, None


def select_fn(x):
    return x[1]


def build_datapipes(root_dir: str, batch_size: int, buffer_size: int):
    file_lister = FileLister([root_dir], recursive=True)
    config_dp, dataset_dp, graph_dp = file_lister.demux(
        3, classify_file_fn, drop_none=True, buffer_size=-1
    )
    config_dp = config_dp.open_files().parse_json_files().map(select_fn)
    graph_dp = graph_dp.open_files().parse_json_files().map(select_fn)
    datapipe = config_dp.zip(dataset_dp).zip(graph_dp).map(map_fn)
    datapipe = datapipe.shuffle().sharding_filter()
    datapipe = Buffer(datapipe, buffer_size=buffer_size, batch_size=batch_size).collate(
        custom_collate
    )

    return datapipe


def classify_file_fn(filepath):
    if filepath.endswith("config.json"):
        return 0
    if filepath.endswith("dataset.h5"):
        return 1
    if filepath.endswith("graph.json"):
        return 2
    return None


def map_fn(x):
    return (x[0][0], x[0][1], x[1])


class Buffer(IterDataPipe):
    def __init__(self, source_datapipe, buffer_size, batch_size):
        self.source_datapipe = source_datapipe
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def get_sample(self, df, idx):
        return torch.tensor(df.iloc[idx]["measurement"], dtype=torch.bool)

    def __iter__(self):
        folder_pairs = list(self.source_datapipe)
        random.shuffle(folder_pairs)  # Shuffle folder pairs for randomness

        for i in range(0, len(folder_pairs), self.buffer_size):
            loaded_data = []
            for j in range(i, min(i + self.buffer_size, len(folder_pairs))):
                config_file, h5_file_path, graph_file = folder_pairs[j]
                logging.info(f"Loading: {h5_file_path}")

                pyg_graph = pyg_graph_data(config_file, graph_file)

                df = pd.read_hdf(h5_file_path, key="data")
                for index, row in df.iterrows():
                    measurement = self.get_sample(df, index)
                    m_onehot = to_one_hot(measurement, 2)
                    _, dim = m_onehot.shape
                    m_shifted_onehot = torch.cat(
                        (torch.zeros(1, dim), m_onehot[:-1]), dim=0
                    )
                    batch = Batch(
                        graph=pyg_graph,
                        m_onehot=m_onehot,
                        m_shifted_onehot=m_shifted_onehot,
                    )
                    loaded_data.append(batch)

            # Shuffle the loaded data for randomness
            random.shuffle(loaded_data)

            # Yield batches of data
            for k in range(0, len(loaded_data), self.batch_size):
                yield loaded_data[k : k + self.batch_size]
