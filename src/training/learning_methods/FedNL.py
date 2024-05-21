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
from src.experiment import Config
from src.training.learning_methods.Strategy import Strategy


class NewtonOptimizer(Optimizer):
	def __init__(self, params: Iterator[Parameter]):
		# Newton method takes no parameters
		defaults = dict()
		super(NewtonOptimizer, self).__init__(params, defaults)

	def step(self, closure: Callable[[torch.Tensor], torch.Tensor] | None = None) -> Optional[float]:
		# The closure function is required to compute the Hessian, if it is not there raise an error
		if closure is None:
			raise RuntimeError('closure function must be provided for NewtonOptimizer')

		model_parameters = [value for param_group in self.param_groups for value in param_group["params"]]

		# Flatten model
		parameter_shapes = [torch.tensor(layer.shape) for layer in model_parameters]
		flat_parameter_shapes = torch.stack([torch.prod(shape) for shape in parameter_shapes])
		cutoff_points = [0, *torch.cumsum(flat_parameter_shapes, dim=0)]
		cutoff_points = list(zip(cutoff_points, cutoff_points[1:]))
		model_parameters = torch.concatenate([value.flatten() for value in model_parameters])

		# Compute the gradient and hessian
		loss = closure(model_parameters)
		gradient = torch.autograd.grad(loss, model_parameters)[0]
		hessian = torch.func.hessian(closure)(model_parameters) + torch.eye(len(gradient)) * 1e-8
		inv_hessian = torch.linalg.inv(hessian)
		update = inv_hessian @ gradient

		# Reshape gradients, hessian and update to its desired shape
		gradients = [
			gradient[start:end].reshape(*parameter_shape)
			for (start, end), parameter_shape in zip(cutoff_points, parameter_shapes)
		]
		hessians = [
			hessian[start:end, start:end].reshape(*parameter_shape, *parameter_shape)
			for (start, end), parameter_shape in zip(cutoff_points, parameter_shapes)
		]
		updates = [
			update[start:end].reshape(*parameter_shape)
			for (start, end), parameter_shape in zip(cutoff_points, parameter_shapes)
		]

		i = 0
		for param_group in self.param_groups:
			for layer in param_group["params"]:
				gradient = gradients[i]
				hessian = hessians[i]
				update = updates[i]
				self.state[layer]["gradients"] = gradient
				self.state[layer]["hessian"] = hessian
				layer.data.sub_(update)
				i += 1


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

		parameter_shapes = [layer.shape for layer in parameters]
		flat_parameter_shapes = [np.prod(shape) for shape in parameter_shapes]
		cutoff_points = [0, *np.cumsum(flat_parameter_shapes)]
		cutoff_points = list(zip(cutoff_points, cutoff_points[1:]))

		# Do local rounds and epochs
		for _ in range(local_rounds):
			for features, labels in train_loader:
				features, labels = features.to(training.DEVICE), labels.to(training.DEVICE)

				def closure(weights):
					# Reshape the weights to the correct shape
					weights = [
						weights[start:end].reshape(parameter_shape)
						for (start, end), parameter_shape in zip(cutoff_points, parameter_shapes)
					]
					net_parameters = {
						name: value for name, value in zip(net.state_dict().keys(), weights)
					}
					prediction = torch.func.functional_call(net, net_parameters, features)

					# Set the parameters as the state dict of the model
					loss = criterion(prediction, labels)
					return loss

				optimizer.zero_grad()
				optimizer.step(closure)

		# Take the gradients and hessian from the optimizer state and transmit the results
		gradients = [value["gradients"] for value in optimizer.state_dict()["state"].values()]
		hessian = [value["hessian"].detach().numpy() for value in optimizer.state_dict()["state"].values()]
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

		counts = [fitres.num_examples for _, fitres in results]

		# Aggregate the gradients
		gradient_results = (parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results)
		gradients = utils.common.compute_weighted_average(gradient_results, counts)

		# Aggregate the hessians
		hessian_results = (fit_res.metrics["hessian"] for _, fit_res in results)
		hessians = utils.common.compute_weighted_average(hessian_results, counts)

		parameter_shapes = [layer.shape for layer in gradients]
		flat_parameter_shapes = [np.prod(shape) for shape in parameter_shapes]
		cutoff_points = [0, *np.cumsum(flat_parameter_shapes)]
		cutoff_points = list(zip(cutoff_points, cutoff_points[1:]))

		gradients = np.concatenate([value.flatten() for value in gradients])
		hessian_template = np.zeros((len(gradients), len(gradients)))
		for i, ((start, end), hessian) in enumerate(zip(cutoff_points, hessians)):
			size = end - start
			hessian = hessian.reshape(size, size)
			hessian_template[start:end, start:end] = hessian
		hessian = hessian_template
		inv_hessian = np.linalg.inv(hessian)
		update = inv_hessian @ gradients
		updates = (
			update[start:end].reshape(*parameter_shape)
			for (start, end), parameter_shape in zip(cutoff_points, parameter_shapes)
		)
		self.current_weights = ndarrays_to_parameters([
			layer - update for layer, update in zip(self.current_weights, updates)
		])
		return self.current_weights, {}

if __name__ == '__main__':
	path = "examples/mnist/logistic_regression/fed_nl.yaml"
	config = Config.from_yaml_file(path)
	data_loader = config.simulation.train_loaders[0]
	model = config.simulation.model
	initial_parameters = training.get_weights(model)

	# Test the initial parameters performance
	loss, accuracy, _ = Strategy.test(initial_parameters, data_loader, config.simulation)
	print(f"Initial loss: {loss}, accuracy: {accuracy}")

	current_weights = initial_parameters
	for _ in range(10):
		gradients, _, metrics = config.simulation.learning_method.train(
			parameters=current_weights,
			train_loader=data_loader,
			simulation=config.simulation,
			metrics={}
		)

		gradients = [np.array(gradient) for gradient in gradients]
		hessians = [np.array(hessian) for hessian in metrics["hessian"]]

		parameter_shapes = [layer.shape for layer in gradients]
		flat_parameter_shapes = [np.prod(shape) for shape in parameter_shapes]
		cutoff_points = [0, *np.cumsum(flat_parameter_shapes)]
		cutoff_points = list(zip(cutoff_points, cutoff_points[1:]))

		gradients = np.concatenate([value.flatten() for value in gradients])
		hessian_template = np.zeros((len(gradients), len(gradients)))
		for i, ((start, end), hessian) in enumerate(zip(cutoff_points, hessians)):
			size = end - start
			hessian = hessian.reshape(size, size)
			hessian_template[start:end, start:end] = hessian
		hessian = hessian_template
		inv_hessian = np.linalg.inv(hessian)
		update = inv_hessian @ gradients
		updates = (
			update[start:end].reshape(*parameter_shape)
			for (start, end), parameter_shape in zip(cutoff_points, parameter_shapes)
		)
		current_weights = [
			layer - update for layer, update in zip(current_weights, updates)
		]

		loss, accuracy, _ = Strategy.test(current_weights, data_loader, config.simulation)
		print(f"Updated loss: {loss}, accuracy: {accuracy}")
