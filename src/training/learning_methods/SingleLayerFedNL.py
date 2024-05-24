from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
import torch
from flwr.common import FitRes, ndarrays_to_parameters, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from numpy import typing as npt
from torch import nn as nn
from torch.nn import functional
from torch.utils.data import DataLoader

from src import training, utils
from src.training.learning_methods.Strategy import Strategy


class SingleLayerFedNL(Strategy):
	def __init__(self, **kwargs):
		super(SingleLayerFedNL, self).__init__(**kwargs)

	@staticmethod
	def get_initial_parameters(model):
		model_weights = [np.zeros(val.shape) for val in model.state_dict().values()]
		initial_parameters = fl.common.ndarrays_to_parameters(model_weights)
		return initial_parameters


	def get_optimizer(self, parameters: Iterator[nn.Parameter]) -> torch.optim.Optimizer:
		# The newton method makes use of our custom newton optimizer
		pass

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
		local_rounds = simulation.local_rounds

		weights, bias = training.get_weights(net)

		model_weights = np.concatenate((np.expand_dims(bias, axis=1), weights), axis=1)
		model_weights = torch.tensor(model_weights).to(training.DEVICE)

		for features, labels in train_loader:
			features, labels = features.to(training.DEVICE), labels.to(training.DEVICE)
			labels = functional.one_hot(labels)

			# Add a bias element to the features
			features = torch.cat([torch.ones_like(features[:, 0:1]), features], dim=1)
			diagonal = features.T @ features

			for _ in range(local_rounds):
				prediction = functional.softmax(features @ model_weights.T)

				hessian = diagonal * torch.sum(prediction.T @ (1 - prediction))
				inverse_hessian = torch.linalg.pinv(hessian)
				gradient = (prediction - labels).T @ features

				model_weights -= gradient @ inverse_hessian

		data_size = len(train_loader.dataset)
		return [gradient.numpy()], data_size, {"hessian": [hessian.numpy()]}

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
		gradients = utils.common.compute_weighted_average(gradient_results, counts)[0]

		# Aggregate the hessians
		hessian_results = (fit_res.metrics["hessian"] for _, fit_res in results)
		hessians = utils.common.compute_weighted_average(hessian_results, counts)[0]

		# Compute the inverse hessian and update the weights
		inverse_hessian = np.linalg.pinv(hessians)
		update = gradients @ inverse_hessian
		current_weights = parameters_to_ndarrays(self.current_weights)
		current_weights[0] -= update[:, 1:]
		current_weights[1] -= update[:, 0]

		self.current_weights = ndarrays_to_parameters(current_weights)
		return self.current_weights, {}
