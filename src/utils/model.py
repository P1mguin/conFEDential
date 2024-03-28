from typing import Callable, Iterator, Type

import torch
import torch.nn as nn

import src.utils as utils


def load_model_from_config(config: dict) -> Type[nn.Module]:
	"""
	Loads a Sequential PyTorch model from a config of a YAML file,
	to describe a model in a YAML file, each layer describes a utility function of torch.nn
	this is followed by the parameters required by the function
	:param config: config containing the model definition
	"""
	model_config = config["model"]
	layers = []
	for layer in model_config["layers"]:
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


def load_optimizer_from_config(config: dict) -> Callable[[Iterator[nn.Parameter]], Type[torch.optim.Optimizer]]:
	"""
	Loads a PyTorch optimizer from a config of a YAML file, the name of the optimizer is followed by its parameters
	:param config: config containing the optimizer definition
	"""
	optimizer_config = config["simulation"]["learning_method"]
	parameters = {key: value for key, value in list(optimizer_config.items())[1:]}
	return lambda params: getattr(torch.optim, optimizer_config["optimizer"])(params, **parameters)
