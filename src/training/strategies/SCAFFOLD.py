from collections import OrderedDict
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from flwr.common import FitRes, ndarrays_to_parameters, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from numpy import typing as npt
from torch import nn as nn
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src import training, utils
from src.training.strategies.Strategy import Strategy
from src.utils.configs import Config


class ScaffoldOptimizer(Optimizer):
	def __init__(self, params: Iterator[Parameter], lr: float = 0.1):
		if lr < 0.:
			raise ValueError(f"Learning rate should be above 0, got {lr}")
		defaults = dict(lr=lr)
		super(ScaffoldOptimizer, self).__init__(params, defaults)

	def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
		loss = None
		if closure is not None:
			with torch.enable_grad():
				loss = closure()

		# For each parameter compute the update
		for group in self.param_groups:
			for p in group["params"]:
				if p.grad is None:
					continue
				grad = p.grad.data

				# Take the global and local direction and compute the model update
				global_c = self.state[p]["global_c"]
				local_c = self.state[p]["local_c"]
				p.data.sub_(group["lr"] * (grad - local_c + global_c))
		return loss


class SCAFFOLD(Strategy):
	def __init__(self, **kwargs) -> None:
		# Set helper variables and hyperparameters
		super(SCAFFOLD, self).__init__(**kwargs)
		self.local_c = None
		self.global_c = None
		self.global_lr = kwargs["global"]["lr"]

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

		# Get the aggregate of the client deltas
		delta_results = [
			(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
			for _, fit_res in results
		]
		deltas_aggregated = utils.common.compute_weighted_average(delta_results)

		# In the first round, global c is uninitialized, do so now
		if self.global_c is None:
			self.global_c = [np.zeros_like(layer) for layer in deltas_aggregated]

		# Get the aggregate of the clients difference with global c
		delta_c_results = [
			(fit_res.metrics["local_c_delta"], fit_res.num_examples)
			for _, fit_res in results
		]
		delta_c_aggregated = utils.common.compute_weighted_average(delta_c_results)

		# Update the current weights and global c
		current_weights = parameters_to_ndarrays(self.current_weights)
		self.current_weights = [
			old_layer + self.global_lr * new_layer
			for old_layer, new_layer in zip(current_weights, deltas_aggregated)
		]
		self.global_c = [
			old_global_c + (len(results) / run_config.get_client_count() * delta_c)
			for old_global_c, delta_c in zip(self.global_c, delta_c_aggregated)
		]

		# Encode and return the current weights, and return the aggregate of the clients directions
		self.current_weights = ndarrays_to_parameters(self.current_weights)
		return self.current_weights, {"global_c": self.global_c}

	def get_optimizer_instance(self, parameters: Iterator[nn.Parameter]) -> torch.optim.Optimizer:
		# SCAFFOLD uses the custom SCAFFOLD optimizer
		return ScaffoldOptimizer(parameters, **self.kwargs["local"])

	def train(
			self,
			parameters: List[npt.NDArray],
			train_loader: DataLoader,
			run_config: Config,
			config: Dict[str, Any]
	) -> Tuple[List[npt.NDArray], int, Dict[str, Any]]:
		# If no local c has been set initialize it at zero
		if self.local_c is None:
			self.local_c = [np.zeros_like(layer) for layer in parameters]

		# If no global c has been received, it is the first round: use zero instead
		if config.get("global_c") is not None:
			global_c = config.get("global_c")
		else:
			global_c = [np.zeros_like(layer) for layer in parameters]

		# Get and set the training configuration
		net = run_config.get_model().to(training.DEVICE)
		if parameters is not None:
			training.set_weights(net, parameters)
		criterion = run_config.get_criterion()
		optimizer = run_config.get_optimizer(net.parameters())
		local_rounds = run_config.get_local_rounds()

		# Get the state of the optimizer and overwrite it with the global and local c
		state_dict = optimizer.state_dict()
		state_dict["state"] = OrderedDict(
			{i: {"local_c": torch.Tensor(local_c_layer), "global_c": torch.Tensor(global_c_layer)}
			 for i, (local_c_layer, global_c_layer) in enumerate(zip(self.local_c, global_c))}
		)
		optimizer.load_state_dict(state_dict)

		# Do local rounds and epochs
		for _ in range(local_rounds):
			for features, labels in train_loader:
				features, labels = features.to(training.DEVICE), labels.to(training.DEVICE)
				optimizer.zero_grad()
				loss = criterion(net(features), labels)
				loss.backward()
				optimizer.step()

		# Compute the difference between the previous model, update the local c and compute the difference between the
		# previous local c and the new local c
		model_delta = [
			new_layer - old_layer for old_layer, new_layer in zip(parameters, training.get_weights(net))
		]
		local_c = [
			(local_c_layer - global_c_layer) + (1 / (local_rounds * self.kwargs["local"]["lr"])) * -model_delta_layer
			for local_c_layer, global_c_layer, model_delta_layer in zip(self.local_c, global_c, model_delta)
		]
		local_c_delta = [
			new_layer - old_layer for new_layer, old_layer in zip(local_c, self.local_c)
		]
		self.local_c = local_c
		data_size = len(train_loader.dataset)

		# Transmit all the differences
		return model_delta, data_size, {"local_c_delta": local_c_delta}
