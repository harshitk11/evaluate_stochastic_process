# utils/config_utils.py

import yaml
from easydict import EasyDict

def load_config(file_path):
    """
    Load configuration from a YAML file and return it in easydict format.
    """
    with open(file_path, 'r') as stream:
        try:
            config_dict = yaml.safe_load(stream)
            return EasyDict(config_dict)
        except yaml.YAMLError as exc:
            raise ValueError(f"Error loading config file {file_path}: {exc}") 
            
def recursive_merge(base_dict, update_dict):
    for key, value in update_dict.items():
        if key in base_dict:
            if isinstance(value, dict):
                recursive_merge(base_dict[key], value)
            else:
                base_dict[key] = value
        else:
            base_dict[key] = value

def merge_configs(base_config, update_config):
    """
    Merge the base configuration with an update configuration.
    Both configs are assumed to be EasyDict objects.
    """
    # Convert EasyDict objects back to dictionaries for merging
    base_dict, update_dict = dict(base_config), dict(update_config)

    # Identify and print updated parameters
    for key, value in update_dict.items():
        if key in base_dict and base_dict[key] != value:
            print(f"Parameter '{key}' updated from {base_dict[key]} to {value}")

    recursive_merge(base_dict, update_dict)
    
    # Convert the merged dictionary back to EasyDict for further dot notation access
    return EasyDict(base_dict)


def save_config(config, file_path):
    """
    Save a configuration (in easydict format) to a YAML file.
    """
    with open(file_path, 'w') as stream:
        try:
            yaml.dump(dict(config), stream)  # Convert EasyDict back to dictionary for dumping
        except yaml.YAMLError as exc:
            print(exc)
            
def dict_recursive(easy_dict_obj):
    """Recursively converts EasyDict objects to regular dictionaries."""
    if isinstance(easy_dict_obj, EasyDict):
        return {k: dict_recursive(v) for k, v in easy_dict_obj.items()}
    elif isinstance(easy_dict_obj, list):
        return [dict_recursive(item) for item in easy_dict_obj]
    else:
        return easy_dict_obj

# Example usage:
# base_config = load_config('./config/base_config.yaml')
# print(base_config.some.nested.property)
