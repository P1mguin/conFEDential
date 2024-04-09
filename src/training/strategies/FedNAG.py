from collections import OrderedDict
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


class FedNAG(Strategy):
	def __init__(self, **kwargs):
		super(FedNAG, self).__init__(**kwargs)

	def aggregate_fit(
			self,
			server_round: int,
			results: List[Tuple[ClientProxy, FitRes]],
			failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
			run_config: Config
	) -> Tuple[Optional[Parameters], Dict[str, Any]]:
		if not results:
			return None, {}

		parameter_results = [
			(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
			for _, fit_res in results
		]

		parameters_aggregated = utils.common.compute_weighted_average(parameter_results)
		parameters_aggregated = ndarrays_to_parameters(parameters_aggregated)

		velocity_results = [
			(fit_res.metrics["velocity"], fit_res.num_examples)
			for _, fit_res in results
		]
		velocity_aggregated = utils.common.compute_weighted_average(velocity_results)

		return parameters_aggregated, {"velocity": velocity_aggregated}

	def get_optimizer_instance(self, parameters: Iterator[nn.Parameter]) -> SGD:
		return torch.optim.SGD(parameters, nesterov=True, **self.kwargs)

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

		if config.get("velocity") is not None:
			state_dict = optimizer.state_dict()
			state_dict["state"] = OrderedDict(
				{i: {"momentum_buffer": torch.Tensor(value)} for i, value in enumerate(config.get("velocity"))}
			)
			optimizer.load_state_dict(state_dict)

		for _ in range(local_rounds):
			for features, labels in train_loader:
				features, labels = features.to(training.DEVICE), labels.to(training.DEVICE)
				optimizer.zero_grad()
				loss = criterion(net(features), labels)
				loss.backward()
				optimizer.step()

		parameters = training.get_weights(net)
		data_size = len(train_loader.dataset)
		velocity = [val["momentum_buffer"].cpu().numpy() for val in optimizer.state.values()]

		return parameters, data_size, {"velocity": velocity}
