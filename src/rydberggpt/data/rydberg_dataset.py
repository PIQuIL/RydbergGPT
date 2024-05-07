import logging
import random
from typing import Tuple

import pandas as pd
import torch
import torchdata  # NOTE: this import is important for the code to work
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.iter import FileLister

from rydberggpt.data.dataclasses import Batch, custom_collate
from rydberggpt.data.utils_graph import pyg_graph_data
from rydberggpt.utils import to_one_hot

logging.basicConfig(level=logging.INFO)


def get_rydberg_dataloader(
    batch_size: int = 10,
    num_workers: int = 0,
    data_path: str = "dataset",
    buffer_size: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    # Tutorial how to use datapipes with Dataloader
    # https://pytorch.org/data/main/dp_tutorial.html
    datapipe = build_datapipes(
        root_dir=data_path, batch_size=batch_size, buffer_size=buffer_size
    )

    data_loader = DataLoader(
        datapipe,
        batch_size=None,
        num_workers=num_workers,
    )

    return data_loader


def build_datapipes(root_dir: str, batch_size: int, buffer_size: int):
    """
    Builds a data pipeline for processing files from a specified directory.

    This function initializes a FileLister to list files from the specified
    directory and its subdirectories. It then demultiplexes the files into
    three separate data pipes for processing configuration, dataset, and
    graph files respectively. The configuration and graph files are opened,
    parsed as JSON, and processed using a custom selection function.
    The data pipes are then zipped together, shuffled, filtered, and buffered
    into batches using a custom collate function.

    Args:
        root_dir (str): The root directory from which to list files.
        batch_size (int): The number of samples per batch.
        buffer_size (int): The buffer size to use when buffering data into batches.

    Returns:
        IterDataPipe: The final data pipe containing batches of processed data.
    """
    file_lister = FileLister([root_dir], recursive=True)
    config_dp, dataset_dp, graph_dp = file_lister.demux(
        3,
        classify_file_fn,
        drop_none=True,
        buffer_size=-1,
    )
    config_dp = config_dp.open_files().parse_json_files()
    graph_dp = graph_dp.open_files().parse_json_files()
    datapipe = config_dp.zip(dataset_dp).zip(graph_dp).map(map_fn)
    datapipe = datapipe.shuffle()
    datapipe = Buffer(source_datapipe=datapipe, buffer_size=buffer_size)
    datapipe = datapipe.batch(batch_size).collate(custom_collate).sharding_filter()

    return datapipe


def classify_file_fn(filepath: str):
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
    def __init__(self, source_datapipe, buffer_size):
        self.source_datapipe = source_datapipe
        self.buffer_size = buffer_size

    def get_sample(self, df, idx):
        return torch.tensor(df.iloc[idx]["measurement"], dtype=torch.bool)

    def __iter__(self):
        folder_pairs = list(self.source_datapipe)

        for i in range(0, len(folder_pairs), self.buffer_size):
            loaded_data = []
            for j in range(i, min(i + self.buffer_size, len(folder_pairs))):
                config_file, h5_file_path, graph_file = folder_pairs[j]

                pyg_graph = pyg_graph_data(config_file[1], graph_file[1])

                df = pd.read_hdf(h5_file_path, key="data")
                for index, _ in df.iterrows():
                    measurement = self.get_sample(df, index)
                    m_onehot = to_one_hot(measurement, 2)
                    loaded_data.append(Batch(graph=pyg_graph, m_onehot=m_onehot))

            # Shuffle the loaded data for randomness
            random.shuffle(loaded_data)

            # Yield batches of data
            for k in range(0, len(loaded_data)):
                yield loaded_data[k]
