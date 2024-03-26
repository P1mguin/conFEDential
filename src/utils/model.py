from typing import Callable, Iterator, Type

import torch
import torch.nn as nn

import src.utils as utils


def load_model_from_yaml_file(yaml_file: str) -> Type[nn.Module]:
	"""
	Loads a Sequential PyTorch model from a YAML file,
	to describe a model in a YAML file, each layer describes a utility function of torch.nn
	this is followed by the parameters required by the function
	:param yaml_file: absolute path to YAML file
	"""
	config = utils.load_yaml_file(yaml_file)["model"]
	layers = []
	for layer in config["layers"]:
		layer_type = getattr(nn, layer["type"])
		parameters = {key: value for key, value in list(layer.items())[1:]}
		layer = layer_type(**parameters)
		layers.append(layer)

	class Net(nn.Module):
		def __init__(self) -> None:
			super(Net, self).__init__()
			self.layers = nn.Sequential(*layers)

		def forward(self, x: torch.Tensor) -> torch.Tensor:
			x = self.layers(x)
			return x

	return Net


def load_optimizer_from_yaml_file(yaml_file: str) -> Callable[[Iterator[nn.Parameter]], Type[torch.optim.Optimizer]]:
	"""
	Loads a PyTorch optimizer from a YAML file, the name of the optimizer is followed by its parameters
	:param yaml_file: absolute path to YAML file
	"""
	config = utils.load_yaml_file(yaml_file)["simulation"]["learning_method"]
	parameters = {key: value for key, value in list(config.items())[1:]}
	return lambda params: getattr(torch.optim, config["optimizer"])(params, **parameters)
