from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
from flwr.common import FitRes, ndarrays_to_parameters, Parameters, parameters_to_ndarrays, Scalar
from flwr.server.client_proxy import ClientProxy
from numpy import typing as npt
from torch import nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from src import training, utils
from src.training.strategies.Strategy import Strategy
from src.utils.configs import Config


class FedAvg(Strategy):
	def __init__(self, **kwargs):
		super(FedAvg, self).__init__(**kwargs)

	def aggregate_fit(
			self,
			server_round: int,
			results: List[Tuple[ClientProxy, FitRes]],
			failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
	) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
		if not results:
			return None, {}

		parameter_results = [
			(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
			for _, fit_res in results
		]
		parameters_aggregated = utils.common.compute_weighted_average(parameter_results)
		parameters_aggregated = ndarrays_to_parameters(parameters_aggregated)

		return parameters_aggregated, {}

	def get_optimizer_instance(self, parameters: Iterator[nn.Parameter]) -> SGD:
		return torch.optim.SGD(parameters, **self.kwargs)

	def train(
			self,
			parameters: List[npt.NDArray],
			train_loader: DataLoader,
			run_config: Config,
			config: Dict[str, Any]
	) -> Tuple[List[npt.NDArray], int, Dict[str, Any]]:
		net = run_config.get_model().to(training.DEVICE)

		if parameters is not None:
			training.set_weights(net, parameters)

		criterion = run_config.get_criterion()
		optimizer = run_config.get_optimizer(net.parameters())
		local_rounds = run_config.get_local_rounds()

		for _ in range(local_rounds):
			for features, labels in train_loader:
				features, labels = features.to(training.DEVICE), labels.to(training.DEVICE)
				optimizer.zero_grad()
				loss = criterion(net(features), labels)
				loss.backward()
				optimizer.step()

		parameters = training.get_weights(net)

		data_size = len(train_loader.dataset)
		return parameters, data_size, {}
