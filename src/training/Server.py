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
from src.experiment import Simulation
from src.training.learning_methods import Strategy


class Server(FedAvg):
	def __init__(self, simulation: Simulation, is_capturing: bool):
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
		self.is_capturing = is_capturing and not os.path.exists(output_directory)

		output_directory = self.simulation.get_capture_directory()

		if self.is_capturing:
			os.makedirs(f"{output_directory}messages/metrics", exist_ok=True)
			os.makedirs(f"{output_directory}aggregates/metrics", exist_ok=True)

	@staticmethod
	def get_evaluate_fn(simulation: Simulation):
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
			self._capture_messages(results)

			output_directory = self.simulation.get_capture_directory()
			self._capture_aggregates(
				parameters_to_ndarrays(aggregated_parameters),
				f"{output_directory}aggregates/aggregates.npz"
			)
			for key, value in config.items():
				self._capture_aggregates(value, f"{output_directory}aggregates/metrics/{key}.npz")

		# Update config with the configuration values received from the aggregation
		self.update_config_fn(config)
		return aggregated_parameters, config

	def update_config_fn(self, config: Dict[str, Any]) -> None:
		# Method that is used to send additional variables to clients
		def fit_config(server_round: int) -> Dict[str, Any]:
			return config

		self.on_fit_config_fn = fit_config

	def _capture_aggregates(self, aggregate: List[npt.NDArray], path: str) -> None:
		# Ensure the path to the file exists
		if not os.path.exists(path):
			# Grab some non-zero message as a representation
			shapes = [layer.shape for layer in aggregate]
			np.savez(path, *[np.zeros((0, *shape)) for shape in shapes])

		# Retrieve the earlier captures of the variable to which this iteration will be appended
		variables = self._load_npz_file(path)

		# Expand the dimension of each layer by one such it supports a new iteration of captures
		for i, (captures, layer) in enumerate(zip(variables, aggregate)):
			shape = captures.shape
			expanded_matrix = np.zeros((shape[0] + 1, shape[1], *shape[2:]))
			expanded_matrix[:-1] = variables[i]
			expanded_matrix[-1] = layer
			variables[i] = expanded_matrix

		# Save this variable
		np.savez(path, *variables)

	def _capture_messages(self, results: List[Tuple[ClientProxy, FitRes]]) -> None:
		output_directory = self.simulation.get_capture_directory()

		# Bundle the messages in a dict
		captured_results = {"parameters": {}, "metrics": {}}
		for client_proxy, fit_results in results:
			cid = int(client_proxy.cid)

			# Store the parameters
			parameters = parameters_to_ndarrays(fit_results.parameters)
			captured_results["parameters"][cid] = parameters

			# Store the metrics
			metrics = fit_results.metrics
			for key, value in metrics.items():
				if not captured_results["metrics"].get(key):
					captured_results["metrics"][key] = {}
				captured_results["metrics"][key][cid] = value

		# Capture the parameters
		parameters_path = f"{output_directory}messages/messages.npz"
		self._capture_variable(captured_results["parameters"], parameters_path)

		# Capture the variables in the metrics
		for key, value in captured_results["metrics"].items():
			capture_path = f"{output_directory}messages/metrics/{key}.npz"
			self._capture_variable(value, capture_path)

	def _capture_variable(self, messages: Dict[int, List[npt.NDArray]], path: str) -> None:
		# Ensure the path to the file exists
		if not os.path.exists(path):
			# Grab some non-zero message as a representation
			message = list(messages.values())[0]
			shapes = [layer.shape for layer in message]
			self._initialize_empty_npz_file(path, shapes)

		# Retrieve the earlier captures of the variable to which this iteration will be appended
		variables = self._load_npz_file(path)

		# Expand the dimension of each layer by one such it supports a new iteration of captures
		for i, layer in enumerate(variables):
			shape = layer.shape
			expanded_matrix = np.zeros((shape[0], shape[1] + 1, *shape[2:]))
			expanded_matrix[:, :-1] = variables[i]
			variables[i] = expanded_matrix

		# Set the value for each client
		for cid, message in messages.items():
			for i, layer in enumerate(message):
				variables[i][cid, -1] = layer

		# Save this variable
		np.savez(path, *variables)

	def _load_npz_file(self, path: str) -> List[npt.NDArray]:
		if not os.path.exists(path):
			raise FileNotFoundError(f"Path {path} does not exist")
		npz_file = np.load(path)
		np_arrays = []
		for file in npz_file.files:
			np_arrays.append(npz_file[file])
		return np_arrays

	def _initialize_empty_npz_file(self, path: str, shapes: List[Tuple[int, ...]]) -> None:
		"""
		Initializes an empty npz file with the given shapes for the amount of clients of the simulation
		:param path: the path to the npz file
		:param shapes: the shapes of the layers to be captured
		"""
		np.savez(path, *[np.zeros((self.simulation.client_count, 0, *shape)) for shape in shapes])
