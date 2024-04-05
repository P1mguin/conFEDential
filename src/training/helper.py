from collections import OrderedDict
from typing import List, Tuple

import numpy.typing as npt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.configs import Config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_weights_from_model(model: nn.Module) -> List[npt.NDArray]:
	"""
	Takes a PyTorch model and returns its parameters as a list of NumPy arrays
	:param model: the PyTorch model
	"""
	return [val.cpu().numpy().copy() for _, val in model.state_dict().items()]


def set_weights(model: nn.Module, weights: List[npt.NDArray]) -> None:
	"""
	Takes a PyTorch model and a list of NumPy arrays and makes the NumPy arrays the weights of the model
	:param model: the PyTorch model
	:param weights: the target weights
	"""
	params_dict = zip(model.state_dict().keys(), weights)
	state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
	model.load_state_dict(state_dict, strict=True)


def train(parameters: List[npt.NDArray], train_loader: DataLoader, config: Config) -> Tuple[List[npt.NDArray], int]:
	"""
	A helper method to train a PyTorch model with a given train loader with a method described in a configuration
	:param parameters: the initial parameters of the model
	:param train_loader: the data to train with
	:param config: the configuration that describes the experiment
	"""
	net = config.get_model().to(DEVICE)

	if parameters is not None:
		set_weights(net, parameters)

	criterion = config.get_criterion()
	optimizer = config.get_optimizer(net.parameters())
	local_rounds = config.get_local_rounds()

	for _ in range(local_rounds):
		for features, labels in train_loader:
			features, labels = features.to(DEVICE), labels.to(DEVICE)
			optimizer.zero_grad()
			loss = criterion(net(features), labels)
			loss.backward()
			optimizer.step()

	parameters = get_weights_from_model(net)

	data_size = len(train_loader.dataset)
	return parameters, data_size


def test(parameters: List[npt.NDArray], test_loader: DataLoader, config: Config) -> Tuple[float, float, int]:
	"""
	A helper method to test a PyTorch model on a given test loader via criteria described in a configuration
	:param parameters: the initial parameters of the model
	:param test_loader: the data to test with
	:param config: the configuration that describes the experiment
	"""
	net = config.get_model().to(DEVICE)

	if parameters is not None:
		set_weights(net, parameters)

	criterion = config.get_criterion()
	correct, total, loss = 0, 0, 0.

	with torch.no_grad():
		for data in test_loader:
			features, labels = data['x'].to(DEVICE), data['y'].to(DEVICE)
			outputs = net(features)
			loss += criterion(outputs, labels).item()
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	accuracy = correct / total

	data_size = len(test_loader.dataset)
	return loss, accuracy, data_size
