import functools
import logging
import os
import resource
import sys
from dataclasses import dataclass, make_dataclass
from typing import Any, Dict, List, Tuple, Type, Union

import torch
import yaml
from torch import nn

logger = logging.getLogger(__name__)


def track_memory_usage(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        # Measure memory used by the function's result
        memory_used = sys.getsizeof(result)

        # Convert bytes into GB, MB, and KB components
        gb, remainder = divmod(memory_used, 1_073_741_824)
        mb, kb = divmod(remainder, 1_048_576)
        usage = f"Memory used by {func.__name__}: {gb} GB, {mb} MB, and {kb} KB"

        logger.info(usage)
        return result

    return wrapper


# def track_memory_usage(func):
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         initial_memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
#         result = func(*args, **kwargs)
#         final_memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
#         memory_difference = final_memory_usage - initial_memory_usage

#         # Convert total KB into GB, MB, and KB components
#         gb, remainder = divmod(memory_difference, 1_048_576)
#         mb, kb = divmod(remainder, 1_024)
#         usage = f"Memory used by {func.__name__}: {gb} GB, {mb} MB, and {kb} KB"

#         logger.info(usage)  # Use memory_logger instead of logger
#         return result

#     return wrapper


def save_to_yaml(data: Dict[str, Any], filename: str) -> None:
    """
    Save a dictionary to a file in YAML format.

    Args:
        data (Dict[str, Any]): The dictionary to be saved.
        filename (str): The path to the file where the dictionary will be saved.

    Returns:
        None
    """
    with open(filename, "w") as file:
        yaml.dump(data, file)


def to_one_hot(
    data: Union[torch.Tensor, List[int], Tuple[int]], num_classes: int
) -> torch.Tensor:
    """
    Converts the input data into one-hot representation.

    Args:
        data: Input data to be converted into one-hot. It can be a 1D tensor, list or tuple of integers.
        num_classes: Number of classes in the one-hot representation.

    Returns:
        data: The one-hot representation of the input data.
    """

    if isinstance(data, (list, tuple)):
        data = torch.tensor(data, dtype=torch.int64)
    elif not isinstance(data, torch.Tensor):
        raise TypeError("Input data must be a tensor, list or tuple of integers.")

    data = nn.functional.one_hot(data.long(), num_classes)

    return data.to(torch.float)


def create_dataclass_from_dict(name: str, data: Dict[str, Any]) -> Type:
    """
    Create a dataclass from a dictionary.

    Args:
        name (str): The name of the dataclass.
        data (Dict[str, Any]): A dictionary containing the dataclass fields and their values.

    Returns:
        Type: A new dataclass with the specified name and fields.
    """
    fields = [(key, type(value)) for key, value in data.items()]
    return make_dataclass(name, fields)


def flatten_yaml(yaml_config: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Flatten a nested YAML configuration dictionary.

    Args:
        yaml_config (Dict[str, Dict[str, Any]]): A nested dictionary representing the YAML configuration.

    Returns:
        Dict[str, Any]: A flattened dictionary with the nested structure removed.
    """
    flattened_config = {}
    for section, section_values in yaml_config.items():
        if isinstance(section_values, dict):
            for key, value in section_values.items():
                flattened_config[f"{key}"] = value
        else:
            flattened_config[section] = section_values
    return flattened_config


def load_yaml_file(path: str, yaml_file_name: str) -> Dict[str, Any]:
    """
    Load the content of a YAML file given its path and file name.

    Args:
        path (str): The path to the directory containing the YAML file.
        yaml_file_name (str): The name of the YAML file.

    Returns:
        Dict[str, Any]: A dictionary containing the YAML content.
    """
    if not yaml_file_name.endswith(".yaml"):
        yaml_file_name += ".yaml"

    yaml_path = os.path.join(path, yaml_file_name)
    with open(yaml_path, "r") as file:
        yaml_content = yaml.safe_load(file)
    return yaml_content


def create_config_from_yaml(yaml_content: Dict) -> dataclass:
    """
    Create a dataclass config object from the given YAML content.

    Args:
        yaml_content (Dict): A dictionary containing the YAML content.

    Returns:
        dataclass: A dataclass object representing the config.
    """
    flattened_config = flatten_yaml(yaml_content)
    Config = create_dataclass_from_dict("Config", flattened_config)
    return Config(**flattened_config)


def load_config_file(checkpoint_path: str, config_file: str = "hparams.yaml") -> str:
    """
    Load the configuration file associated with a given checkpoint.

    Args:
        checkpoint_path (str): The path to the checkpoint file.
        config_file (str, optional): The name of the configuration file, defaults to "hparams.yaml".

    Returns:
        str: The path to the configuration file.

    Raises:
        FileNotFoundError: If the configuration file is not found in the specified directory.
    """
    config_dir = os.path.dirname(os.path.dirname(checkpoint_path))

    if not os.path.exists(os.path.join(config_dir, config_file)):
        raise FileNotFoundError(f"No config file found in {config_dir}")

    return os.path.join(config_dir, config_file)
