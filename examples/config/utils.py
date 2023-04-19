from dataclasses import asdict, dataclass, replace

import yaml

from config.base_config import Config


def create_config_from_yaml(yaml_path: str) -> Config:
    with open(yaml_path, "r") as file:
        yaml_config = yaml.safe_load(file)

    # Flatten the nested dictionary
    flattened_config = {}
    for section, section_values in yaml_config.items():
        for key, value in section_values.items():
            flattened_config[key] = value

    # Create a new Config object with values from the YAML file
    return Config(**flattened_config)
