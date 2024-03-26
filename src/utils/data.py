from typing import Callable, List, Tuple

from torch.utils.data import DataLoader

import src.utils as utils
from src import federated_datasets


def load_dataloaders_from_config(config: dict) -> Tuple[List[DataLoader], DataLoader]:
	"""
	Loads non-iid data loaders from YAML file for training and testing.
	The amount of training data loaders is equal to the amount of clients specified in the yaml file,
	the testing happens at the server and so only one data loader is returned with all testing data.
	No validation data loader is returned
	:param config: Dictionary of the simulation configuration
	"""
	client_count = config["simulation"]["client_count"]
	batch_size = config["simulation"]["batch_size"]
	preprocess_fn = utils.load_preprocess_func_from_function_string(config["dataset"]["preprocess_fn"])
	alpha = config["dataset"]["splitter"]["alpha"]
	percent_noniid = config["dataset"]["splitter"]["percent_noniid"]
	dataclass = getattr(federated_datasets, config["dataset"]["name"])
	return dataclass.load_data(client_count, batch_size, preprocess_fn, alpha, percent_noniid)


def load_preprocess_func_from_function_string(function_string: str) -> Callable[[dict], dict]:
	"""
	Loads and parses a python function described as string
	:param function_string: python function with correct indentation
	"""
	namespace = {}
	exec(function_string, namespace)
	return namespace["preprocess_fn"]
