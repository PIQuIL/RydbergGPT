from dataclasses import make_dataclass
from typing import Any, Dict, Type

import yaml


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
