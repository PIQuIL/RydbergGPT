import json
import os

from typing import Tuple

import h5py
import networkx as nx
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from rydberggpt.data.dataclasses import Batch, custom_collate
from rydberggpt.data.loading.utils import read_subfolder_data
from rydberggpt.data.utils_graph import networkx_to_pyg_data
from rydberggpt.utils import to_one_hot


def get_rydberg_dataloader(
    batch_size: int = 32,
    test_size: float = 0.2,
    num_workers: int = 0,
    data_path: str = "data",
) -> Tuple[DataLoader, DataLoader]:
    df, graph_data = read_subfolder_data(data_path=data_path)
    # check that df is not empty
    assert not df.empty, "Dataframe is empty. Check that the data_path is correct."
    # train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    train_dataset = RydbergDataset(df, graph_data)
    val_dataset = RydbergDataset(df, graph_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate,
    )
    return train_loader, val_loader


class RydbergDataset(Dataset):
    """
    A custom Dataset class for handling Rydberg atom data.

    Attributes:
        dataframe (pd.DataFrame): The input dataframe containing Rydberg atom data.
    """

    def __init__(self, dataframe, graph_data):
        """
        Initialize the RydbergDataset with the given dataframe.

        Args:
            dataframe (pd.DataFrame): The input dataframe containing Rydberg atom data.
        """
        self.dataframe = dataframe
        self.graph_data = graph_data

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Batch:
        """
        Get the Batch object at the specified index.

        Args:
            index (int): The index of the Batch object in the dataset.

        Returns:
            Batch: A Batch object containing the data for the specified index.
        """
        row = self.dataframe.iloc[index]
        graph_dict = self.graph_data[index]  # dict
        graph_nx = nx.node_link_graph(graph_dict)

        node_features = torch.tensor(
            [row["delta"], row["omega"], row["beta"], row["Rb"]],
            dtype=torch.float32,
        )
        pyg_graph = networkx_to_pyg_data(graph_nx, node_features)

        measurements = torch.tensor(row["measurement"], dtype=torch.int64)
        m_onehot = to_one_hot(measurements, 2)  # because Rydberg states are 0 or 1

        _, dim = m_onehot.shape
        m_shifted_onehot = torch.cat((torch.zeros(1, dim), m_onehot[:-1]), dim=0)
        return Batch(
            graph=pyg_graph,
            m_onehot=m_onehot,
            m_shifted_onehot=m_shifted_onehot,
        )
