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
from src.training.learning_methods.Strategy import Strategy


class NewtonOptimizer(Optimizer):
	def __init__(self, params: Iterator[Parameter]):
		# Newton method takes no parameters
		defaults = dict()
		super(NewtonOptimizer, self).__init__(params, defaults)

	def step(self, closure: Callable[[Tuple[torch.Tensor]], torch.Tensor] | None = None) -> Optional[float]:
		# The closure function is required to compute the Hessian, if it is not there raise an error
		if closure is None:
			raise RuntimeError('closure function must be provided for NewtonOptimizer')

		model_parameters = tuple(value for param_group in self.param_groups for value in param_group["params"])

		# The gradients are directly computed from the closure function
		loss = closure(*model_parameters)
		gradients = torch.autograd.grad(loss, model_parameters)

		# Ensure the gradients are at least 2D with shape (out_feature, in_feature), the tuple in the list is
		# (gradient, is_expanded). We need the boolean to correct the expansion later
		gradients = [
			(gradient.unsqueeze(-1), True) if gradient.ndim == 1 else (gradient, False)
			for gradient in gradients
		]

		# The hessian is computed using torch func
		hessians = torch.func.hessian(closure, argnums=tuple(range(len(model_parameters))))(*model_parameters)
		hessians = (hessian[i] for i, hessian in enumerate(hessians))

		# Expand the hessians similarly, do not keep track of the boolean as that is in the gradients
		hessians = (
			hessian.unsqueeze(-2).unsqueeze(-1) if is_expanded else hessian
			for (_, is_expanded), hessian in zip(gradients, hessians)
		)

		# Reshape the matrix to be square, the hessian at index i, j is hessian[i, :, j]
		hessians = (torch.stack([hessian[i, :, i] for i in range(hessian.size(0))]) for hessian in hessians)

		# Use a generator to limit memory usage
		def compute_inverse_hessian(hessian):
			for out_feature_hessian in hessian:
				try:
					inv_hessian = torch.linalg.inv(out_feature_hessian)
					inv_hessian[inv_hessian.isnan()] = 0.
				except torch.linalg.LinAlgError:
					inv_hessian = torch.ones_like(out_feature_hessian)
				yield inv_hessian

		# Compute the update per layer
		for i in range(len(gradients)):
			gradient, is_expanded = gradients[i]
			hessian = next(hessians)
			inverse_hessian = compute_inverse_hessian(hessian)
			update = torch.stack([
				out_feature_inverse_hessian @ out_feature_gradient
				for out_feature_inverse_hessian, out_feature_gradient in zip(inverse_hessian, gradient)
			])

			# Correct the expansion
			if is_expanded:
				hessian = hessian.view(-1)
				gradient = gradient.view(-1)
				update = update.view(-1)

			# Set the state with the gradient and hessian, and update the model parameters
			self.state[model_parameters[i]]["gradients"] = gradient.detach()
			self.state[model_parameters[i]]["hessian"] = hessian.detach()
			model_parameters[i].data.sub_(update)

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

				def get_gradient(*weights):
					# Query the model as if the parameters had been set
					net_parameters = {name: value for name, value in zip(net.state_dict().keys(), weights)}
					prediction = torch.func.functional_call(net, net_parameters, features)

					# Set the parameters as the state dict of the model
					loss = criterion(prediction, labels)
					return loss

				optimizer.zero_grad()
				optimizer.step(get_gradient)

		# Take the gradients and hessian from the optimizer state and transmit the results
		gradients = [value["gradients"] for value in optimizer.state_dict()["state"].values()]
		hessian = [value["hessian"] for value in optimizer.state_dict()["state"].values()]
		data_size = len(train_loader.dataset)
		return gradients, data_size, {"hessian": hessian}

	def aggregate_fit(
			self,
			server_round: int,
			results: List[Tuple[ClientProxy, FitRes]],
			failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
			simulation
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

		gradients = [
			(np.expand_dims(gradient, axis=-1), True) if gradient.ndim == 1 else (gradient, False)
			for gradient in gradients
		]
		hessians = [
			np.expand_dims(hessian, axis=(-2, -1)) if is_expanded else hessian
			for (_, is_expanded), hessian in zip(gradients, hessians)
		]

		inverse_hessians = []
		for hessian in hessians:
			inverse_hessian = []
			for out_feature_hessian in hessian:
				try:
					inv_hessian = np.linalg.inv(out_feature_hessian)
				except np.linalg.LinAlgError:
					inv_hessian = np.ones_like(out_feature_hessian)
				inverse_hessian.append(inv_hessian)
			inverse_hessians.append(np.stack(inverse_hessian))

		# Per layer calculate the model update
		current_weights = parameters_to_ndarrays(self.current_weights)
		for i in range(len(gradients)):
			gradient, is_expanded = gradients[i]
			inverse_hessian = inverse_hessians[i]
			update = np.stack([inverse_hessian[j] @ gradient[j] for j in range(len(gradient))])

			# Correct the expansion
			if is_expanded:
				update = update.reshape(-1)

			current_weights[i] -= update

		self.current_weights = ndarrays_to_parameters(current_weights)
		return self.current_weights, {}
