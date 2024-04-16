import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
import numpy.typing as npt
from flwr.common import FitRes, Parameters, parameters_to_ndarrays, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from src.utils.configs import Config


def get_capturing_strategy(
		run_config: Config,
		evaluate_fn: Callable[[int, List[npt.NDArray], Dict[str, Scalar]], Tuple[float, Dict[str, Scalar]]],
		is_capturing: bool = False
) -> fl.server.strategy.Strategy:
	"""
	Generates a flower learning strategy in which the transmitted parameters
	are captured in a numpy file in the .captured folder
	:param run_config: the loaded yaml config
	:param evaluate_fn: the evaluation function to use on the server level
	:param is_capturing: whether to maintain the captured parameters in a numpy file
	"""
	# Get the strategy and set the initial parameters of the strategy model
	strategy = run_config.get_strategy()
	strategy.set_parameters(run_config.get_initial_parameters())

	class FedCapture(FedAvg):
		def __init__(self) -> None:
			(
				fraction_evaluate,
				fraction_fit,
				min_available_clients,
				min_evaluate_clients,
				min_fit_clients
			) = run_config.get_client_selection_config()

			# Initialize the flower client
			initial_parameters = run_config.get_initial_parameters()
			super(FedCapture, self).__init__(
				fraction_fit=fraction_fit,
				fraction_evaluate=fraction_evaluate,
				min_fit_clients=min_fit_clients,
				min_evaluate_clients=min_evaluate_clients,
				min_available_clients=min_available_clients,
				evaluate_fn=evaluate_fn,
				initial_parameters=initial_parameters
			)

			# Set capturing parameters
			self.config = {}
			self.client_count = run_config.get_client_count()
			self.output_directory = run_config.get_output_capture_directory_path()
			if is_capturing:
				self._initialize_directory_path()

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

		def _capture_results(self, results: List[Tuple[ClientProxy, FitRes]]) -> None:
			# Package the results in a dict
			captured_results = {"parameters": {}, "metrics": {}}
			for client_proxy, fit_results in results:
				cid = int(client_proxy.cid)
				parameters = parameters_to_ndarrays(fit_results.parameters)
				metrics = fit_results.metrics

				captured_results["parameters"][cid] = parameters
				for key, value in metrics.items():
					if not captured_results["metrics"].get(key):
						captured_results["metrics"][key] = {}
					captured_results["metrics"][key][cid] = value

			# Capture the parameters
			parameters_path = f"{self.output_directory}parameters.npz"
			self._capture_variable(captured_results["parameters"], parameters_path)

			# Capture the variables in the metrics
			for key, value in captured_results["metrics"].items():
				self._initialize_directory_path(has_metrics=True)
				capture_path = f"{self.output_directory}metrics/{key}.npz"
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

		def _initialize_directory_path(self, has_metrics: bool = False) -> None:
			# Ensure all directories to the output file exist by creating them if they do not yet exist
			if has_metrics:
				os.makedirs(os.path.dirname(f"{self.output_directory}metrics/"), exist_ok=True)
				os.makedirs(os.path.dirname(f"{self.output_directory}aggregates/metrics/"), exist_ok=True)
			else:
				os.makedirs(os.path.dirname(f"{self.output_directory}"), exist_ok=True)
				os.makedirs(os.path.dirname(f"{self.output_directory}aggregates/"), exist_ok=True)

		def _initialize_empty_npz_file(self, path: str, shapes: List[Tuple[int, ...]]) -> None:
			"""
			Initializes an empty npz file with the given shapes for the amount of clients of the simulation
			:param path: the path to the npz file
			:param shapes: the shapes of the layers to be captured
			"""
			np.savez(path, *[np.zeros((self.client_count, 0, *shape)) for shape in shapes])

		def aggregate_fit(
				self,
				server_round: int,
				results: List[Tuple[ClientProxy, FitRes]],
				failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
		) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
			# Call the actual learning method
			aggregated_parameters, config = strategy.aggregate_fit(server_round, results, failures, run_config)

			# Capture the results
			if is_capturing:
				self._capture_results(results)
				self._capture_aggregates(parameters_to_ndarrays(aggregated_parameters),
										 f"{self.output_directory}aggregates/parameters.npz")
				for key, value in config.items():
					self._capture_aggregates(value, f"{self.output_directory}aggregates/metrics/{key}.npz")

			# Update config with the configuration values received from the aggregation
			self.update_config_fn(config)
			return aggregated_parameters, config

		def update_config_fn(self, config: Dict[str, Any]) -> None:
			# Method that is used to send additional variables to clients
			def fit_config(server_round: int) -> Dict[str, Any]:
				return config
			self.on_fit_config_fn = fit_config

	return FedCapture()
