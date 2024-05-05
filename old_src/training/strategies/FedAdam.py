from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from flwr.common import FitRes, ndarrays_to_parameters, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from numpy import typing as npt
from torch import nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from src import training, utils
from src.training.strategies.Strategy import Strategy
from src.utils.old_configs import Config


class FedAdam(Strategy):
	def __init__(self, **kwargs) -> None:
		# Super call
		super(FedAdam, self).__init__(**kwargs)

		# Set the variables used for learing
		self.first_momentum = None
		self.second_momentum = None

		# Set the hyperparameters
		self.global_lr = kwargs["global"]["lr"]
		self.betas = kwargs["global"]["betas"]
		self.eps = kwargs["global"]["eps"]
		self.weight_decay = kwargs["global"]["weight_decay"]

	def aggregate_fit(
			self,
			server_round: int,
			results: List[Tuple[ClientProxy, FitRes]],
			failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
			run_config: Config
	) -> Tuple[Optional[Parameters], Dict[str, Any]]:
		# If no results have been received return nothing
		if not results:
			return None, {}

		# Get the current weight and decode them
		current_weights = parameters_to_ndarrays(self.current_weights)

		# Aggregate and negate the parameters
		delta_results = [
			(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
			for _, fit_res in results
		]
		deltas_aggregated = utils.common.compute_weighted_average(delta_results)

		# In the first round the momenta are unitialized
		if self.first_momentum is None:
			self.first_momentum = [np.zeros_like(layer) for layer in deltas_aggregated]
		if self.second_momentum is None:
			self.second_momentum = [np.zeros_like(layer) for layer in deltas_aggregated]

		# Calculate the new momenta
		self.first_momentum = [
			self.betas[0] * x + (1 - self.betas[0]) * y
			for x, y in zip(self.first_momentum, deltas_aggregated)
		]
		self.second_momentum = [
			self.betas[1] * x + (1 - self.betas[1]) * (y * y)
			for x, y in zip(self.second_momentum, deltas_aggregated)
		]

		# Update and return the new weights based on the momenta
		self.current_weights = [
			x + self.global_lr * (y / (np.sqrt(z) + self.eps))
			for x, y, z in zip(current_weights, self.first_momentum, self.second_momentum)
		]
		self.current_weights = ndarrays_to_parameters(self.current_weights)
		return self.current_weights, {}

	def get_optimizer_instance(self, parameters: Iterator[nn.Parameter]) -> SGD:
		# FedAdam works with SGD at the client
		return torch.optim.SGD(parameters, **self.kwargs["local"])

	def train(
			self,
			parameters: List[npt.NDArray],
			train_loader: DataLoader,
			run_config: Config,
			config: Dict[str, Any]
	) -> Tuple[List[npt.NDArray], int, Dict[str, Any]]:
		# Get and set training configuration
		net = run_config.get_model().to(training.DEVICE)
		if parameters is not None:
			training.set_weights(net, parameters)
		criterion = run_config.get_criterion()
		optimizer = run_config.get_optimizer(net.parameters())
		local_rounds = run_config.get_local_rounds()

		# Do local rounds and epochs
		for _ in range(local_rounds):
			for features, labels in train_loader:
				features, labels = features.to(training.DEVICE), labels.to(training.DEVICE)
				optimizer.zero_grad()
				loss = criterion(net(features), labels)
				loss.backward()
				optimizer.step()

		# Compute and return the difference between the initial model and the new model
		gradient = [
			new_layer - old_layer for old_layer, new_layer in zip(parameters, training.get_weights(net))
		]
		data_size = len(train_loader.dataset)
		return gradient, data_size, {}
