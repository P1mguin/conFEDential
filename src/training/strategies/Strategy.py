from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union

import numpy.typing as npt
import torch
import torch.nn as nn
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from torch.utils.data import DataLoader

import src.training as training
from src.utils.configs import Config


class Strategy(ABC):
	def __init__(self, **kwargs):
		self.kwargs = kwargs
		self.current_weights = None

	@abstractmethod
	def get_optimizer_instance(self, parameters: Iterator[nn.Parameter]) -> torch.optim.Optimizer:
		"""
		Returns the optimizer that will be used for this training strategy
		:param parameters: the parameters that will be used in the optimizer
		"""
		pass

	@abstractmethod
	def train(
			self,
			parameters: List[npt.NDArray],
			train_loader: DataLoader,
			run_config: Config,
			config: Dict[str, Any]
	) -> Tuple[List[npt.NDArray], int, Dict[str, Any]]:
		"""
		A method to train a PyTorch model with a given train loader with a method described in a configuration
		:param parameters: the initial parameters of the model
		:param train_loader: the data to train with
		:param run_config: the configuration that describes the experiment
		:param config: extra parameters used for learning
		"""
		pass

	@abstractmethod
	def aggregate_fit(
			self,
			server_round: int,
			results: List[Tuple[ClientProxy, FitRes]],
			failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
	) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
		"""
		A method to aggregate the results of the client of the global training round
		:param server_round: an indicator of which global iterations the system is in
		:param results: the results of the clients
		:param failures: the clients that failed
		"""
		pass

	@staticmethod
	def test(parameters: List[npt.NDArray], test_loader: DataLoader, config: Config) -> Tuple[float, float, int]:
		"""
		A helper method to test a PyTorch model on a given test loader via criteria described in a configuration
		:param parameters: the initial parameters of the model
		:param test_loader: the data to test with
		:param config: the configuration that describes the experiment
		"""
		net = config.get_model().to(training.DEVICE)

		if parameters is not None:
			training.set_weights(net, parameters)

		criterion = config.get_criterion()
		correct, total, loss = 0, 0, 0.

		with torch.no_grad():
			for data in test_loader:
				features, labels = data['x'].to(training.DEVICE), data['y'].to(training.DEVICE)
				outputs = net(features)
				loss += criterion(outputs, labels).item()
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
		accuracy = correct / total

		data_size = len(test_loader.dataset)
		return loss, accuracy, data_size

	def set_parameters(self, parameters: Parameters):
		self.current_weights = parameters
