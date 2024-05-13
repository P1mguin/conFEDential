from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy.typing as npt
import torch
import torch.nn as nn
from flwr.common import FitRes, Parameters
from flwr.server.client_proxy import ClientProxy
from torch.utils.data import DataLoader

import src.training as training


class Strategy(ABC):
	def __init__(self, **kwargs) -> None:
		self.kwargs = kwargs
		self.current_weights = None

	@abstractmethod
	def get_optimizer(self, parameters: Iterator[nn.Parameter]) -> torch.optim.Optimizer:
		"""
		Returns the optimizer that will be used for this training strategy given the parameters of the model
		"""
		pass

	@abstractmethod
	def train(
			self,
			parameters: List[npt.NDArray],
			train_loader: DataLoader,
			simulation,
			metrics: Dict[str, Any]
	) -> Tuple[List[npt.NDArray], int, Dict[str, Any]]:
		"""
		A method to train a PyTorch model with a given train loader with a method described in a configuration
		:param parameters: the initial parameters of the model
		:param train_loader: the data to train with
		:param simulation: the configuration that describes the local learning
		:param metrics: extra parameters used for learning
		"""
		pass

	@abstractmethod
	def aggregate_fit(
			self,
			server_round: int,
			results: List[Tuple[ClientProxy, FitRes]],
			failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
			simulation
	) -> Tuple[Optional[Parameters], Dict[str, Any]]:
		"""
		A method to aggregate the results of the client of the global training round
		:param server_round: an indicator of which global iterations the system is in
		:param results: the results of the clients
		:param failures: the clients that failed
		:param simulation: the configuration that describes the experiment
		"""
		pass

	@staticmethod
	def test(
			parameters: List[npt.NDArray] | None,
			test_loader: DataLoader,
			simulation
	) -> Tuple[float, float, int]:
		"""
		A helper method to test a PyTorch model on a given test loader via criteria described in a configuration
		:param parameters: the initial parameters of the model
		:param test_loader: the data to test with
		:param simulation: the configuration that describes the experiment
		"""
		# Get and set the testing configuration
		net = simulation.model.to(training.DEVICE)
		if parameters is not None:
			training.set_weights(net, parameters)
		criterion = simulation.criterion
		correct, total_loss = 0, 0.

		# Disable gradient calculation for testing
		with torch.no_grad():
			for features, labels in test_loader:
				features, labels = features.to(training.DEVICE), labels.to(training.DEVICE)
				outputs = net(features)

				# Accumulate the total loss
				total_loss += criterion(outputs, labels).item()

				# Get the amount of correct predictions
				_, predicted = torch.max(outputs.data, 1)
				correct += (predicted == labels).sum().item()

		# Compute the accuracy and return the performance
		data_size = len(test_loader.dataset)
		accuracy = correct / data_size
		loss = total_loss / data_size

		return loss, accuracy, data_size

	def set_parameters(self, parameters: Parameters) -> None:
		"""
		Sets the parameters that the aggregation is working with
		:param parameters: the new parameters
		"""
		self.current_weights = parameters
