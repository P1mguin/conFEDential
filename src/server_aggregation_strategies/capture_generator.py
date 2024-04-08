import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
import numpy.typing as npt
from flwr.common import FitRes, Parameters, parameters_to_ndarrays, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from src.utils.configs import Config


def get_capturing_strategy(
		config: Config,
		evaluate_fn: Callable[[int, List[npt.NDArray], Dict[str, Scalar]], Tuple[float, Dict[str, Scalar]]],
		is_capturing: bool = False
) -> fl.server.strategy.Strategy:
	"""
	Generates a flower learning strategy in which the transmitted parameters
	are captured in a numpy file in the .captured folder
	:param config: the loaded yaml config
	:param evaluate_fn: the evaluation function to use on the server level
	:param is_capturing: whether to maintain the captured parameters in a numpy file
	"""
	strategy = config.get_strategy()

	class FedCapture(FedAvg):
		def __init__(self) -> None:
			(
				fraction_evaluate,
				fraction_fit,
				min_available_clients,
				min_evaluate_clients,
				min_fit_clients
			) = config.get_client_selection_config()

			initial_parameters = config.get_initial_parameters()
			super(FedCapture, self).__init__(
				fraction_fit=fraction_fit,
				fraction_evaluate=fraction_evaluate,
				min_fit_clients=min_fit_clients,
				min_evaluate_clients=min_evaluate_clients,
				min_available_clients=min_available_clients,
				evaluate_fn=evaluate_fn,
				initial_parameters=initial_parameters
			)

			self.client_count = config.get_client_count()
			self.output_path = config.get_output_capture_file_path()
			if is_capturing:
				self._initialize_directory_path()

		def _initialize_directory_path(self) -> None:
			"""
			Ensures all directories to the output file exist by creating them if they do not yet exist
			"""
			os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

		def _capture_parameters(self, captured_parameters: Dict[int, List[npt.NDArray]]) -> None:
			"""
			Capture the parameters of the clients transmitted to the server
			:param captured_parameters: the parameters transmitted to the server
			"""
			captures = self._load_npz_file(captured_parameters)

			# Expand the dimension of each client by one
			for i in range(len(captures)):
				shape = captures[i].shape
				expanded_matrix = np.zeros((shape[0], shape[1] + 1, *shape[2:]))
				expanded_matrix[:, :-1] = captures[i]
				captures[i] = expanded_matrix

			# For each client set the value
			for client, parameters in captured_parameters.items():
				for i, layer in enumerate(parameters):
					captures[i][client, -1] = layer

			np.savez(self.output_path, *captures)

		def _load_npz_file(self, captured_parameters: Dict[int, List[npt.NDArray]]) -> List[npt.NDArray]:
			"""
			Loads the already captured parameters, initializes empty npz file as a list of
			shape (number of layers (including biases), number clients, 0 (number of iterations), layer shape)
			:param captured_parameters: A list with each element the intercepted layer per client per iterations
			"""
			if not os.path.exists(self.output_path):
				# The shape of captured parameter should account for several clients,
				# for several iterations with the shape of the layer
				shapes = [np.zeros((self.client_count, 0, *layer.shape)) for layer in
						  list(captured_parameters.values())[0]]
				np.savez(self.output_path, *shapes)

			npz_file = np.load(self.output_path)
			previous_captures = []
			for file in npz_file.files:
				previous_captures.append(npz_file[file])
			return previous_captures

		def aggregate_fit(
				self,
				server_round: int,
				results: List[Tuple[ClientProxy, FitRes]],
				failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
		) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
			if is_capturing:
				captured_parameters = {}
				for client_proxy, fit_results in results:
					cid = int(client_proxy.cid)
					parameters = parameters_to_ndarrays((fit_results.parameters))
					captured_parameters[cid] = parameters
				self._capture_parameters(captured_parameters)

			return strategy.aggregate_fit(server_round, results, failures)

	return FedCapture()
