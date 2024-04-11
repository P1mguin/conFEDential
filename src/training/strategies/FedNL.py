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


class NewtonOptimizer(Optimizer):
	def __init__(self, params: Iterator[Parameter]):
		defaults = dict()
		super(NewtonOptimizer, self).__init__(params, defaults)

	def step(self, closure = None) -> Optional[float]:
		if closure is None:
			raise RuntimeError('closure function must be provided for NewtonOptimizer')

		loss = closure()
		parameters = [value for value in self.param_groups[0]["params"]]
		loss_gradient = torch.autograd.grad(loss, parameters, create_graph=True)
		if loss is None:
			raise RuntimeError('closure function did not return the loss value')

		for parameter, gradients in zip(parameters, loss_gradient):
			is_unsqueezed = gradients.ndim == 1
			if is_unsqueezed:
				gradients = gradients.unsqueeze(0)

			# Compute the hessian
			hessians = torch.zeros(*gradients.shape, gradients.shape[-1])
			for i in range(gradients.size(0)):
				for j in range(gradients.size(1)):
					hessians[:, j] = torch.autograd.grad(gradients[i][j], parameter, create_graph=True)[0]

			# Per parameter compute the update
			updates = torch.zeros_like(gradients)
			for i in range(gradients.size(0)):
				# Compute the inverse hessian
				try:
					inv_hessian = torch.linalg.inv(hessians[i])
				except torch.linalg.LinAlgError:
					inv_hessian = torch.ones_like(hessians[i])

				# Compute the update and set in list
				update = inv_hessian @ gradients[i]
				updates[i] = update

			if is_unsqueezed:
				hessians = hessians.view(hessians.shape[1:])
				gradients = gradients.view(-1)
				updates = updates.view(-1)

			self.state[parameter]["gradients"] = gradients
			self.state[parameter]["hessian"] = hessians
			parameter.data.sub_(updates)
		return loss.item()


class FedNL(Strategy):
	def __init__(self, **kwargs):
		super(FedNL, self).__init__(**kwargs)

	def aggregate_fit(
			self,
			server_round: int,
			results: List[Tuple[ClientProxy, FitRes]],
			failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
			run_config: Config
	) -> Tuple[Optional[Parameters], Dict[str, Any]]:
		if not results:
			return None, {}

		gradient_results = [
			(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
			for _, fit_res in results
		]
		gradients = utils.common.compute_weighted_average(gradient_results)

		hessian_results = [
			(fit_res.metrics["hessian"], fit_res.num_examples)
			for _, fit_res in results
		]
		hessians = utils.common.compute_weighted_average(hessian_results)

		inverse_hessians = [np.zeros_like(layer) for layer in hessians]
		for i, layer in enumerate(hessians):
			for j, output in enumerate(layer):
				try:
					inverse_hessian = np.linalg.inv(output)
				except np.linalg.LinAlgError:
					inverse_hessian = np.zeros_like(output)
				inverse_hessians[i][j] = inverse_hessian

		# Per parameter compute the update
		updates = []
		for i, (gradient, inverse_hessian) in enumerate(zip(gradients, inverse_hessians)):
			updates.append(np.zeros_like(gradient))
			if inverse_hessian.ndim == 2:
				updates[i] = gradient @ inverse_hessian
				continue

			for j, (parameter_gradient, parameter_inverse_hessian) in enumerate(zip(gradient, inverse_hessian)):
				updates[i][j] = parameter_gradient @ parameter_inverse_hessian

		current_weights = parameters_to_ndarrays(self.current_weights)
		self.current_weights = [
			layer - update
			for layer, update in zip(current_weights, updates)
		]
		self.current_weights = ndarrays_to_parameters(self.current_weights)

		return self.current_weights, {}

	def get_optimizer_instance(self, parameters: Iterator[nn.Parameter]) -> torch.optim.Optimizer:
		return NewtonOptimizer(parameters)

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

				def closure(*args, **kwargs):
					optimizer.zero_grad()
					loss = criterion(net(features), labels)
					return loss

				optimizer.step(closure)

		gradients = [value["gradients"].detach().numpy() for value in optimizer.state_dict()["state"].values()]
		hessian = [value["hessian"].detach().numpy() for value in optimizer.state_dict()["state"].values()]

		data_size = len(train_loader.dataset)

		return gradients, data_size, {"hessian": hessian}
