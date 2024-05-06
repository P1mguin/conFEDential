from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

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
from src.experiment import Simulation
from src.training.learning_methods.Strategy import Strategy


class NewtonOptimizer(Optimizer):
	def __init__(self, params: Iterator[Parameter]):
		# Newton method takes no parameters
		defaults = dict()
		super(NewtonOptimizer, self).__init__(params, defaults)

	def step(self, closure=None) -> Optional[float]:
		# The closure function is required to compute the Hessian, if it is not there raise an error
		if closure is None:
			raise RuntimeError('closure function must be provided for NewtonOptimizer')

		# Compute the loss gradient based on the loss and current parameters
		loss = closure()
		if loss is None:
			raise RuntimeError('closure function did not return the loss value')
		parameters = [value for value in self.param_groups[0]["params"]]
		loss_gradient = torch.autograd.grad(loss, parameters, create_graph=True)

		# Compute the update per layer
		for parameter, gradients in zip(parameters, loss_gradient):
			# We assume each layer to work on multiple variables, if that is not the case, expand the gradient
			# to mimic it working on one variable
			is_unsqueezed = gradients.ndim == 1
			if is_unsqueezed:
				gradients = gradients.unsqueeze(0)

			# Compute the hessian per parameter
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
					# When the elements become infinitesimally small, the inverse sometimes contains nans
					inv_hessian[inv_hessian.isnan()] = 0.
				except torch.linalg.LinAlgError:
					inv_hessian = torch.ones_like(hessians[i])

				# Compute the update and set in list
				update = inv_hessian @ gradients[i]
				updates[i] = update

			# Correct the expansion
			if is_unsqueezed:
				hessians = hessians.view(hessians.shape[1:])
				gradients = gradients.view(-1)
				updates = updates.view(-1)

			# Set the state with the gradient and hessian, and update the model parameters
			self.state[parameter]["gradients"] = gradients.detach()
			self.state[parameter]["hessian"] = hessians.detach()
			parameter.data.sub_(updates)
		# Return the loss
		return loss.item()


class FedNL(Strategy):
	def __init__(self, **kwargs):
		super(FedNL, self).__init__(**kwargs)

	def get_optimizer(self, parameters: Iterator[nn.Parameter]) -> torch.optim.Optimizer:
		# The newton method makes use of our custom newton optimizer
		return NewtonOptimizer(parameters)

	def train(
			self,
			parameters: List[npt.NDArray],
			train_loader: DataLoader,
			simulation: Simulation,
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

				# Define the closure function that returns the loss of the model
				def closure(*args, **kwargs):
					optimizer.zero_grad()
					loss = criterion(net(features), labels)
					return loss

				optimizer.step(closure)

		# Take the gradients and hessian from the optimizer state and transmit the results
		gradients = [value["gradients"].numpy() for value in optimizer.state_dict()["state"].values()]
		hessian = [value["hessian"].numpy() for value in optimizer.state_dict()["state"].values()]
		data_size = len(train_loader.dataset)
		return gradients, data_size, {"hessian": hessian}

	def aggregate_fit(
			self,
			server_round: int,
			results: List[Tuple[ClientProxy, FitRes]],
			failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
			simulation: Simulation
	) -> Tuple[Optional[Parameters], Dict[str, Any]]:
		# If no results have been received, return noting
		if not results:
			return None, {}

		# Aggregate the gradients
		gradient_results = [
			(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
			for _, fit_res in results
		]
		gradients = utils.common.compute_weighted_average(gradient_results)

		# Aggregate the hessians
		hessian_results = [
			(fit_res.metrics["hessian"], fit_res.num_examples)
			for _, fit_res in results
		]
		hessians = utils.common.compute_weighted_average(hessian_results)

		# Per layer calculate the model update
		updates = []
		for hessian, gradient in zip(hessians, gradients):
			# We assume the layer to want to predict multiple parameters, if it is only for one parameter the gradient is
			# one dimensional, then expand the gradient.
			is_unsqueezed = gradient.ndim == 1
			if is_unsqueezed:
				gradient = np.expand_dims(gradient, 0)
				hessian = np.expand_dims(hessian, 0)

			# Per parameter compute the update
			layer_updates = np.zeros_like(gradient)
			for i, (parameter_hessian, parameter_gradient) in enumerate(zip(hessian, gradient)):
				# Compute the inverse hessian
				try:
					inv_hessian = np.linalg.inv(parameter_hessian)
				except np.linalg.LinAlgError:
					inv_hessian = np.ones_like(parameter_hessian)

				# Set the update for the parameter
				layer_updates[i] = inv_hessian @ parameter_gradient

			# Correct the earlier gradient expansion; Only the layer update is relevant
			if is_unsqueezed:
				layer_updates = layer_updates.squeeze(0)

			updates.append(layer_updates)

		# Update the model, encode it and return it
		current_weights = parameters_to_ndarrays(self.current_weights)
		self.current_weights = [
			layer - update
			for layer, update in zip(current_weights, updates)
		]
		self.current_weights = ndarrays_to_parameters(self.current_weights)
		return self.current_weights, {}
