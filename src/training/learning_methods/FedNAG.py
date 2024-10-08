import copy
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


class FedNAG(Strategy):
	def __init__(self, **kwargs):
		super(FedNAG, self).__init__(**kwargs)

	def get_optimizer(self, parameters: Iterator[nn.Parameter]) -> SGD:
		# FedNAG makes use of SGD with Nesterov momentum, the friction is assumed to be in the key word arguments
		return torch.optim.SGD(parameters, nesterov=True, **self.kwargs)

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

		# If a velocity has been received by the server use that as the starting point
		if metrics.get("velocity") is not None:
			# Get the state of the optimizer and overwrite the current momentum
			state_dict = optimizer.state_dict()
			state_dict["state"] = {i: {"momentum_buffer": torch.Tensor(value.copy())} for i, value in
								   enumerate(metrics.get("velocity"))}
			optimizer.load_state_dict(state_dict)

		# Do local rounds and epochs
		for _ in range(local_rounds):
			for features, labels in train_loader:
				features, labels = features.to(training.DEVICE), labels.to(training.DEVICE)
				optimizer.zero_grad()
				loss = criterion(net(features), labels)
				loss.backward()
				optimizer.step()

		# Get the latest model parameters and velocity from the net and transmit them
		parameters = training.get_weights(net)
		data_size = len(train_loader.dataset)
		velocity = [val["momentum_buffer"].cpu().numpy() for val in optimizer.state.values()]
		return parameters, data_size, {"velocity": velocity}

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

		# Aggregate the results and encode them
		counts = [fitres.num_examples for _, fitres in results]
		parameter_results = (parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results)
		parameters_aggregated = utils.common.compute_weighted_average(parameter_results, counts)
		parameters_aggregated = ndarrays_to_parameters(parameters_aggregated)

		# Aggregate the velocities
		velocity_results = (fit_res.metrics["velocity"] for _, fit_res in results)
		velocity_aggregated = utils.common.compute_weighted_average(velocity_results, counts)
		velocity_aggregated = [layer for layer in velocity_aggregated]

		# Return the aggregated results and velocities
		return parameters_aggregated, {"velocity": velocity_aggregated}

	def compute_metric_updates(self, **kwargs):
		state_dicts = copy.deepcopy(kwargs.get("state_dicts"))
		metrics = copy.deepcopy(kwargs.get("metrics"))
		features = kwargs.get("features")
		labels = kwargs.get("labels")
		simulation = kwargs.get("simulation")
		if metrics.get("velocity") is None:
			return None

		# Find the parameters in the Nesterov loss
		friction = self.kwargs.get("friction", 0.9)
		state_dict_keys = state_dicts[0].keys()
		metric_layers = [key for key in state_dict_keys if state_dicts[0][key].requires_grad]
		for i, metric_layer in enumerate(metric_layers):
			updates = metrics["velocity"][i] * friction
			for j, update in enumerate(updates):
				state_dicts[j][metric_layer] = torch.tensor(
					state_dicts[j][metric_layer] + update, requires_grad=True, device=training.DEVICE
				)

		template_model = simulation.model
		criterion = simulation.criterion
		criterion.reduction = "none"

		is_singular_batch = features.size(0) == 1
		if is_singular_batch:
			features = torch.cat([features, features])

		predictions = torch.stack([
			torch.func.functional_call(template_model, state_dict, features)
			for state_dict in state_dicts
		])

		if is_singular_batch:
			predictions = torch.stack([global_round_value[0].unsqueeze(0) for global_round_value in predictions])

		losses = torch.stack([criterion(prediction, labels) for prediction in predictions])

		gradients = []
		parameter_keys = [name for name, _ in template_model.named_parameters()]
		for i, state_dict in enumerate(state_dicts):
			model_gradients = []
			for loss in losses[i]:
				loss.backward(retain_graph=True)
				model_gradients.append((state_dict[key].grad for key in parameter_keys))
			gradients.append((torch.stack(layers) for layers in zip(*model_gradients)))
		gradients = (torch.stack(layers, dim=1) for layers in zip(*gradients))

		learning_rate = self.kwargs.get("lr", 0.001)
		metrics["velocity"] = [
			friction * velocity - learning_rate * gradient
			for velocity, gradient in zip(metrics["velocity"], gradients)
		]

		# Expand the tensors to 5D
		metrics["velocity"] = [
			torch.stack([utils.reshape_to_4d(x, batched=True) for x in layer]) for layer in metrics["velocity"]
		]

		return metrics
