import collections
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
import numpy.typing as npt
import wandb
from flwr.common import FitRes, Parameters, parameters_to_ndarrays, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from src import training
from src.training.learning_methods import Strategy


class Server(FedAvg):
	def __init__(self, simulation, is_capturing: bool):
		fraction_evaluate = 0.
		min_evaluate_clients = 0
		fraction_fit = simulation.fraction_fit
		min_available_clients = max(int(simulation.client_count * fraction_fit), 1)

		model = simulation.model
		model_weights = training.get_weights(model)
		initial_parameters = fl.common.ndarrays_to_parameters(model_weights)

		evaluate_fn = Server.get_evaluate_fn(simulation)
		super(Server, self).__init__(
			fraction_fit=fraction_fit,
			fraction_evaluate=fraction_evaluate,
			min_fit_clients=min_available_clients,
			min_evaluate_clients=min_evaluate_clients,
			min_available_clients=min_available_clients,
			evaluate_fn=evaluate_fn,
			initial_parameters=initial_parameters
		)

		# Set capturing parameters
		self.config = {}
		self.simulation = simulation

		output_directory = simulation.get_capture_directory()
		self.is_capturing = is_capturing and not os.path.exists(f"{output_directory}aggregates")

		output_directory = self.simulation.get_capture_directory()

		if self.is_capturing:
			os.makedirs(f"{output_directory}messages/metrics", exist_ok=True)
			os.makedirs(f"{output_directory}aggregates/metrics", exist_ok=True)

	@staticmethod
	def get_evaluate_fn(simulation):
		test_loader = simulation.test_loader

		def evaluate(
				server_round: int,
				parameters: fl.common.NDArrays,
				metrics: Dict[str, Scalar]
		) -> Tuple[float, Dict[str, Scalar]]:
			loss, accuracy, data_size = Strategy.test(parameters, test_loader, simulation)
			wandb.log({"loss": loss, "accuracy": accuracy})
			return loss, {"accuracy": accuracy, "data_size": data_size}

		return evaluate

	def aggregate_fit(
			self,
			server_round: int,
			results: List[Tuple[ClientProxy, FitRes]],
			failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
	) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
		# Call the actual learning method
		learning_method = self.simulation.learning_method
		aggregated_parameters, config = learning_method.aggregate_fit(server_round, results, failures, self.simulation)

		# Capture the results
		if self.is_capturing:
			self._capture_results(server_round, results, aggregated_parameters, config)

		# Update config with the configuration values received from the aggregation
		self.update_config_fn(config)
		return aggregated_parameters, config

	def update_config_fn(self, config: Dict[str, Any]) -> None:
		# Method that is used to send additional variables to clients
		def fit_config(server_round: int) -> Dict[str, Any]:
			return config

		self.on_fit_config_fn = fit_config

	def _capture_results(
			self,
			server_round: int,
			results: List[Tuple[ClientProxy, FitRes]],
			aggregated_parameters: Parameters,
			metrics: dict
	) -> None:
		self._capture_aggregates(server_round, aggregated_parameters, metrics)
		self._capture_messages(server_round, results)

	def _capture_messages(self, server_round, messages: List[Tuple[ClientProxy, FitRes]]) -> None:
		base_path = self.simulation.get_capture_directory()
		base_path = f"{base_path}messages/"

		participating_clients = [int(client_proxy.cid) for client_proxy, _ in messages]
		client_count = self.simulation.client_count

		# Function that converts the received messages to the format (client_count, iteration, *shape)
		# and fills in the blanks for the clients that did not participate in the round
		def reshape_to_federation_capture(messages):
			# Stack the layers together
			messages = list(map(np.stack, zip(*messages)))

			# Create a list of zeros for the clients that did not participate and set the variables for those that did
			results = [
				np.expand_dims(np.zeros_like(message[0]), axis=0).repeat(client_count, axis=0) for message in messages
			]
			for result, message in zip(results, messages):
				result[participating_clients] = message

			return results

		# Take the parameters, transpose them and stack them
		parameter_messages = [parameters_to_ndarrays(fitres.parameters) for _, fitres in messages]
		parameter_messages = reshape_to_federation_capture(parameter_messages)
		self._capture_variable(
			server_round, parameter_messages, f"{base_path}messages.npz", is_message=True, is_aggregate=True
		)

		# Combine the metrics in one dict
		metrics = collections.defaultdict(list)
		for _, fitres in messages:
			for key, value in fitres.metrics.items():
				metrics[key].append(value)
		metrics = dict(metrics)

		for key, value in metrics.items():
			value = reshape_to_federation_capture(value)
			self._capture_variable(server_round, value, f"{base_path}metrics/{key}.npz", is_message=True)

	def _capture_aggregates(self, server_round: int, aggregated_parameters: Parameters, metrics: dict):
		base_path = self.simulation.get_capture_directory()
		base_path = f"{base_path}aggregates/"

		# Convert the aggregated parameters to a list of numpy arrays
		aggregated_parameters = parameters_to_ndarrays(aggregated_parameters)
		self._capture_variable(server_round, aggregated_parameters, f"{base_path}aggregates.npz", is_aggregate=True)

		# Capture all metrics
		for key, value in metrics.items():
			self._capture_variable(server_round, value, f"{base_path}metrics/{key}.npz")

	def _capture_variable(
			self,
			server_round: int,
			values: List[npt.NDArray],
			path: str,
			is_aggregate: bool = False,
			is_message: bool = False
	):
		# The axis along which the expansion can happen to account for several iterations
		expansion_axis = 1 if is_message else 0

		# If nothing has been captured yet, the file needs to be initialized
		# The initial parameter is all zeros for the metric and the initial model parameters for an aggregate
		if server_round == 1:
			if is_aggregate:
				saved_values = training.get_weights(self.simulation.model)

				# For messages repeat the messages for each client
				if is_message:
					client_count = self.simulation.client_count
					saved_values = [
						np.expand_dims(saved_value, axis=0).repeat(client_count, axis=0)
						for saved_value in saved_values
					]
			else:
				saved_values = [np.zeros_like(value) for value in values]

			# Expand everything to account for several iterations
			saved_values = [np.expand_dims(value, expansion_axis) for value in saved_values]
		else:
			# Load in the saved variable
			saved_values = np.load(path)
			saved_values = [saved_values[file] for file in saved_values.files]

		values = [np.expand_dims(value, expansion_axis) for value in values]
		values = [
			np.concatenate((saved_value, value), axis=expansion_axis)
			for value, saved_value in zip(values, saved_values)
		]
		np.savez_compressed(path, *values)
