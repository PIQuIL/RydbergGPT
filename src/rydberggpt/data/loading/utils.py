import json
import os
from typing import Dict, List, Tuple

import pandas as pd
import torch


def contains_invalid_numbers(tensor):
    return torch.isnan(tensor).any() or torch.isinf(tensor).any()


def read_subfolder_data(data_path: str = "data") -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Read data from subfolders within the specified data_path and return a DataFrame and a list of graphs.

    This function reads the configuration, dataset, and graph files from each subfolder within the specified
    data_path. It constructs a DataFrame containing the configuration and measurement data, and a list of
    graph dictionaries.

    Args:
        data_path (str, optional): The path to the parent folder containing the subfolders with data.
                                   Default value is "data".

    Returns:
        Tuple[pd.DataFrame, List[Dict]]: A tuple containing a DataFrame with the configuration and measurement
                                          data, and a list of graph dictionaries.
    """

    l_dirs = [
        d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))
    ]

    data = []
    list_of_graphs = []
    for l_dir in l_dirs:
        chunked_dataset_dirs = [
            d
            for d in os.listdir(os.path.join(data_path, l_dir))
            if os.path.isdir(os.path.join(data_path, l_dir, d))
        ]
        for chunked_dataset_dir in chunked_dataset_dirs:
            folder_path = os.path.join(data_path, l_dir, chunked_dataset_dir)
            if os.path.isdir(folder_path):
                with open(os.path.join(folder_path, "config.json"), "r") as f:
                    config = json.load(f)
                df = pd.read_hdf(os.path.join(folder_path, "dataset.h5"), key="data")
                graph_dict_from_json = read_graph_from_json(
                    os.path.join(folder_path, "graph.json")
                )

                for _, row in df.iterrows():
                    data.append(
                        {
                            **config,
                            "measurement": row["measurement"],
                        }
                    )
                    list_of_graphs.append(graph_dict_from_json)

    return pd.DataFrame(data), list_of_graphs


def read_graph_from_json(file_path: str) -> Dict:
    """
    Read a JSON file and convert it to a dictionary representing a NetworkX graph.

    Args:
        file_path: Path to the JSON file to read.

    Returns:
        A dictionary representing a NetworkX graph.
    """
    with open(file_path, "r") as f:
        graph_dict = json.load(f)
    return graph_dict
