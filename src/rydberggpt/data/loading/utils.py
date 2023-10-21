import json
import os
from typing import Dict, List, Tuple

import pandas as pd
import torch


def contains_invalid_numbers(tensor):
    return torch.isnan(tensor).any() or torch.isinf(tensor).any()


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
