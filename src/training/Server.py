import collections
import itertools
import os
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import flwr as fl
import h5py
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
			messages = zip(*messages)

			# Create a list of zeros for the clients that did not participate and set the variables for those that did
			def get_expanded_message(message):
				for i in range(client_count):
					if i in participating_clients:
						index = next(j for j in range(len(participating_clients)) if participating_clients[j] == i)
						yield next(itertools.islice(message, index, None))
					else:
						yield np.zeros_like(message[0])

			messages = (get_expanded_message(message) for message in messages)
			return messages

		# Take the parameters, transpose them and stack them
		parameter_messages = (parameters_to_ndarrays(fitres.parameters) for _, fitres in messages)
		parameter_messages = reshape_to_federation_capture(parameter_messages)
		value_shapes = self.simulation.model_config.get_gradient_shapes()
		value_shapes = (tuple(layer_shape) for value_shape in value_shapes for layer_shape in value_shape)
		self._capture_variable(
			server_round,
			parameter_messages,
			f"{base_path}messages.hdf5",
			is_message=True,
			is_parameters=True,
			value_shapes=value_shapes
		)

		# Combine the metrics in one dict
		metrics = collections.defaultdict(list)
		for _, fitres in messages:
			for key, value in fitres.metrics.items():
				metrics[key].append(value)
		metrics = dict(metrics)

		for key, value in metrics.items():
			value_shapes = (layer.shape for layer in value[0])
			value = reshape_to_federation_capture(value)
			self._capture_variable(
				server_round, value, f"{base_path}metrics/{key}.hdf5", is_message=True, value_shapes=value_shapes
			)

	def _capture_aggregates(self, server_round: int, aggregated_parameters: Parameters, metrics: dict):
		base_path = self.simulation.get_capture_directory()
		base_path = f"{base_path}aggregates/"

		# Convert the aggregated parameters to a list of numpy arrays
		aggregated_parameters = parameters_to_ndarrays(aggregated_parameters)
		value_shapes = self.simulation.model_config.get_gradient_shapes()
		value_shapes = (tuple(layer_shape) for value_shape in value_shapes for layer_shape in value_shape)
		self._capture_variable(
			server_round,
			aggregated_parameters,
			f"{base_path}aggregates.hdf5",
			is_parameters=True,
			value_shapes=value_shapes
		)

		# Capture all metrics
		for key, value in metrics.items():
			value = [layer for layer in value]
			value_shapes = (layer.shape for layer in metrics[key])
			self._capture_variable(
				server_round, value, f"{base_path}metrics/{key}.hdf5", value_shapes=value_shapes
			)

	def _capture_variable(
			self,
			server_round: int,
			values: Generator[npt.NDArray, None, None] | List[npt.NDArray],
			path: str,
			is_parameters: bool = False,
			is_message: bool = False,
			value_shapes: Optional[List[Tuple[int]]] = None
	):
		# If nothing has been captured yet, the file needs to be initialized
		# The initial parameter is all zeros for the metric and the initial model parameters for an aggregate
		if server_round == 1:
			# Add the global round before the value shapes
			global_rounds = self.simulation.global_rounds + 1
			value_shapes = (
				(global_rounds, *value_shape) for value_shape in value_shapes
			)

			# If it is a message capture, add the client to the shape
			client_count = self.simulation.client_count
			if is_message:
				value_shapes = (
					(client_count, *value_shape) for value_shape in value_shapes
				)

			# Create an hdf5 file with the correct shape and fill with variables
			with h5py.File(path, 'w') as hf:
				for i, value_shape in enumerate(value_shapes):
					hf.create_dataset(str(i), shape=value_shape, chunks=True, dtype=np.float32)

			# If it is a parameter capture, set the initial model parameters
			if is_parameters:
				initial_parameters_values = training.get_weights(self.simulation.model)
				with h5py.File(path, 'r+') as hf:
					for i, initial_layer in enumerate(initial_parameters_values):
						if is_message:
							hf[str(i)][:, 0] = initial_layer
						else:
							hf[str(i)][0] = initial_layer

		# Open the saved HDF5 file in read-write mode
		with h5py.File(path, 'r+') as hf:
			for i, value in enumerate(values):
				if is_message:
					for j, client_value in enumerate(value):
						hf[str(i)][j, server_round] = client_value
				else:
					hf[str(i)][server_round] = value
