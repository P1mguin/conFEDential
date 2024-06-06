from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
from flwr.common import FitRes, ndarrays_to_parameters, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from numpy import typing as npt
from torch import nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from src import training, utils
from src.training.learning_methods.Strategy import Strategy


class FedAvg(Strategy):
	def __init__(self, **kwargs):
		super(FedAvg, self).__init__(**kwargs)

	def get_optimizer(self, parameters: Iterator[nn.Parameter]) -> SGD:
		# FedAVG uses SGD at the client
		return torch.optim.SGD(parameters, **self.kwargs)

	def train(
			self,
			parameters: List[npt.NDArray],
			train_loader: DataLoader,
			simulation,
			metrics: Dict[str, Any]
	) -> Tuple[List[npt.NDArray], int, Dict[str, Any]]:
		# Get and set training configuration
		net = simulation.model.to(training.DEVICE)
		if parameters is not None:
			training.set_weights(net, parameters)
		criterion = simulation.criterion
		optimizer = simulation.get_optimizer(net.parameters())
		local_rounds = simulation.local_rounds

		# Do local rounds and epochs
		for _ in range(local_rounds):
			for features, labels in train_loader:
				features, labels = features.to(training.DEVICE), labels.to(training.DEVICE)
				optimizer.zero_grad()
				loss = criterion(net(features), labels)
				loss.backward()
				optimizer.step()

		# Get the latest model parameters from the net and transmit them
		parameters = training.get_weights(net)
		data_size = len(train_loader.dataset)
		return parameters, data_size, {}

	def aggregate_fit(
			self,
			server_round: int,
			results: List[Tuple[ClientProxy, FitRes]],
			failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
			simulation
	) -> Tuple[Optional[Parameters], Dict[str, Any]]:
		# If no results have been received return nothing
		if not results:
			return None, {}

		# Aggregate and return the results
		counts = [fitres.num_examples for _, fitres in results]
		parameter_results = (parameters_to_ndarrays(fit_res.parameters)	for _, fit_res in results)
		parameters_aggregated = utils.common.compute_weighted_average(parameter_results, counts)
		parameters_aggregated = ndarrays_to_parameters(parameters_aggregated)
		return parameters_aggregated, {}
