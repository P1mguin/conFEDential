from typing import Callable, List, Tuple, Type

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src import federated_datasets


def load_dataloaders(config: dict) -> Tuple[List[DataLoader], DataLoader]:
	client_count = config["simulation"]["client_count"]
	batch_size = config["simulation"]["batch_size"]
	preprocess_fn = load_func(config["dataset"]["preprocess_fn"])
	alpha = config["dataset"]["splitter"]["alpha"]
	percent_noniid = config["dataset"]["splitter"]["percent_noniid"]
	dataclass = getattr(federated_datasets, config["dataset"]["name"])
	return dataclass.load_data(client_count, batch_size, preprocess_fn, alpha, percent_noniid)


def load_yaml_file(yaml_file: str) -> str:
	with open(yaml_file, 'r') as f:
		return yaml.safe_load(f)


def load_func(function_string: str) -> Callable[[dict], dict]:
	namespace = {}
	exec(function_string, namespace)
	return namespace['preprocess_fn']


def load_model(yaml_file: str) -> Type[nn.Module]:
	config = load_yaml_file(yaml_file)['model']
	layers = []
	for layer in config['layers']:
		layer_type = getattr(nn, layer['type'])
		parameters = {key: value for key, value in list(layer.items())[1:]}
		layer = layer_type(**parameters)
		layers.append(layer)

	class Net(nn.Module):
		def __init__(self):
			super(Net, self).__init__()
			self.layers = nn.Sequential(*layers)

		def forward(self, x: torch.Tensor) -> torch.Tensor:
			x = self.layers(x)
			return x

	return Net


def load_optimizer(yaml_file: str) -> Type[torch.optim.Optimizer]:
	config = load_yaml_file(yaml_file)['simulation']['learning_method']
	parameters = {key: value for key, value in list(config.items())[1:]}
	return lambda params: getattr(torch.optim, config['optimizer'])(params, **parameters)
