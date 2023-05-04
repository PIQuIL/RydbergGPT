from dataclasses import make_dataclass
from typing import Any, Dict, List, Tuple, Type, Union

import numpy as np
import torch
import yaml
from torch import nn


def to_one_hot(
    data: Union[torch.Tensor, List[int], Tuple[int]], num_classes: int
) -> torch.Tensor:
    """
    Converts the input data into one-hot representation.

    Args:
    - data: Input data to be converted into one-hot. It can be a 1D tensor, list or tuple of integers.
    - num_classes: Number of classes in the one-hot representation.

    Returns:
    - data: The one-hot representation of the input data.
    """

    if isinstance(data, (list, tuple)):
        data = torch.tensor(data, dtype=torch.int64)
    elif not isinstance(data, torch.Tensor):
        raise TypeError("Input data must be a tensor, list or tuple of integers.")

    data = nn.functional.one_hot(data, num_classes)

    return data.to(torch.float)


def create_dataclass_from_dict(name: str, data: Dict[str, Any]) -> Type:
    fields = [(key, type(value)) for key, value in data.items()]
    return make_dataclass(name, fields)


def flatten_yaml(yaml_config: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    flattened_config = {}
    for section, section_values in yaml_config.items():
        for key, value in section_values.items():
            flattened_config[f"{key}"] = value
    return flattened_config


def create_config_from_yaml(yaml_path: str):
    with open(yaml_path, "r") as file:
        yaml_config = yaml.safe_load(file)

    flattened_config = flatten_yaml(yaml_config)
    Config = create_dataclass_from_dict("Config", flattened_config)

    return Config(**flattened_config)
