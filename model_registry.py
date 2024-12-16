import os
import importlib
from inspect import isclass
from torch.utils.data import Dataset
from torch.nn import Module

# Global dataset registry
MODEL_REGISTRY = {}


def register_models():
    """
    Dynamically registers all dataset classes in the data directory.

    Args:
        data_path (str): The base Python path to the dataset module.
    """
    # Get the list of Python files in the data directory
    data_paths = ["custom_models", "transformer"]
    for data_path in data_paths:
        data_dir = os.path.dirname(__file__) + "/" + data_path
        dataset_files = [
            f for f in os.listdir(data_dir) if f.endswith(".py") and not f.startswith("__")
        ]

        for file in dataset_files:
            module_name = f"{data_path}.{file[:-3]}"  # Convert file name to module path
            module = importlib.import_module(module_name)

            # Register all dataset classes in the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isclass(attr) and issubclass(attr, Module) and attr != Module:
                    MODEL_REGISTRY[attr_name] = attr


def get_model(name, **kwargs):
    """
    Fetch a dataset class by name and initialize it.

    Args:
        name (str): Name of the dataset class.
        kwargs: Arguments to initialize the dataset.

    Returns:
        torch.utils.data.Dataset: An instance of the dataset class.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in the registry.")
    return MODEL_REGISTRY[name](**kwargs)
