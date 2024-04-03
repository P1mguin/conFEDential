from collections import OrderedDict
from typing import Callable, Iterator, List, Tuple, Type

import numpy.typing as npt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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


def train(
		epochs: int,
		parameters: List[npt.NDArray],
		model_class: Type[nn.Module],
		train_loader: DataLoader,
		criterion_class: Type[nn.Module],
		optimizer_class: Callable[[Iterator[nn.Parameter]], Type[torch.optim.Optimizer]]
) -> Tuple[List[npt.NDArray], int]:
	"""
	Helper function that trains the model in an isolated function scope
	:param epochs: number of epochs
	:param parameters: initial parameters of the model
	:param model_class: class of the model
	:param train_loader: data loader containing the data for the batches
	:param criterion_class: class of the criterion
	:param optimizer_class: callable method that summons the optimizer given the model parameters
	"""
	net = model_class().to(DEVICE)

	if parameters is not None:
		set_weights(net, parameters)

	criterion = criterion_class()
	optimizer = optimizer_class(net.parameters())

	for _ in range(epochs):
		for features, labels in train_loader:
			features, labels = features.to(DEVICE), labels.to(DEVICE)
			optimizer.zero_grad()
			loss = criterion(net(features), labels)
			loss.backward()
			optimizer.step()

	parameters = get_weights_from_model(net)

	data_size = len(train_loader.dataset)
	return parameters, data_size


def test(
		parameters: List[npt.NDArray],
		model_class: Type[nn.Module],
		test_loader: DataLoader,
		criterion_class: Type[nn.Module]
) -> Tuple[float, float, int]:
	"""
	Helper function that tests the model in an isolated function scope
	:param parameters: initial parameters of the model
	:param model_class: class of the model
	:param test_loader: data loader containing the data for the batches
	:param criterion_class: class of the criterion
	"""
	net = model_class().to(DEVICE)

	if parameters is not None:
		set_weights(net, parameters)

	criterion = criterion_class()
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
