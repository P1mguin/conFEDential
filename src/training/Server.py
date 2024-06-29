import collections
import gc
import os
import sys
from logging import DEBUG
from typing import Any, Dict, List, Optional, Tuple, Union

import flwr as fl
import h5py
import numpy as np
import numpy.typing as npt
import torch
import wandb
from flwr.common import FitRes, Parameters, parameters_to_ndarrays, Scalar, log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

import src.utils as utils
from src import training
from src.training.learning_methods import Strategy


class Server(FedAvg):
	def __init__(self, simulation, is_capturing: bool):
		fraction_evaluate = 0.
		min_evaluate_clients = 0
		fraction_fit = simulation.fraction_fit
		min_available_clients = max(int(simulation.client_count * fraction_fit), 1)

		initial_parameters = simulation.model_config.get_initial_parameters()

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
		memory_usage = get_objects_memory_usage()
		log(
			DEBUG,
			[(x[0], f"{x[1] / 1024 / 1024}MB") for x in list(reversed(sorted(memory_usage, key=lambda x: x[1])))][:1000]
		)
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
		self._capture_aggregates(server_round, results, aggregated_parameters, metrics)
		self._capture_messages(server_round, results)

	def _capture_messages(self, server_round, messages: List[Tuple[ClientProxy, FitRes]]) -> None:
		base_path = self.simulation.get_capture_directory()
		base_path = f"{base_path}messages/"

		parameter_messages = {
			client_proxy.cid: parameters_to_ndarrays(fitres.parameters) for client_proxy, fitres in messages
		}
		self._capture_message_round(server_round, parameter_messages, f"{base_path}messages.hdf5")

		# Combine the metrics in one dict
		metrics = collections.defaultdict(dict)
		for client_proxy, fitres in messages:
			for key, value in fitres.metrics.items():
				metrics[key][client_proxy.cid] = value

		for key, value in metrics.items():
			self._capture_message_round(server_round, value, f"{base_path}metrics/{key}.hdf5")

	def _capture_aggregates(
			self,
			server_round: int,
			results: List[Tuple[ClientProxy, FitRes]],
			aggregated_parameters: Parameters,
			metrics: dict
	):
		base_path = self.simulation.get_capture_directory()
		base_path = f"{base_path}aggregates/"

		# Convert the aggregated parameters to a list of numpy arrays
		aggregated_parameters = parameters_to_ndarrays(aggregated_parameters)
		self._capture_aggregate_round(
			server_round,
			aggregated_parameters,
			f"{base_path}aggregates.hdf5",
			is_parameters=True
		)

		# Add the message metrics that are not in the aggregated metrics
		non_included_metric_keys = results[0][1].metrics.keys() - metrics.keys()
		counts = [fitres.num_examples for _, fitres in results]
		non_aggregate_metrics = {
			key: [fitres.metrics[key] for _, fitres in results] for key in non_included_metric_keys
		}
		non_aggregate_metrics = {
			key: utils.common.compute_weighted_average(non_aggregate_metrics[key], counts)
			for key in non_included_metric_keys
		}

		# Capture all metrics
		for key, value in {**metrics, **non_aggregate_metrics}.items():
			value = [layer.cpu() if isinstance(layer, torch.Tensor) else layer for layer in value]
			self._capture_aggregate_round(server_round, value, f"{base_path}metrics/{key}.hdf5")

	def _capture_message_round(
			self,
			server_round: int,
			values: Dict[int, List[npt.NDArray]],
			path: str,
	):
		# If nothing has been captured yet, the file needs to be initialized
		# The initial parameter is all zeros for the metric and the initial model parameters for an aggregate
		if server_round == 1:
			# Create the values that are put in the hdf5 file initial row, they are always 0
			value_shapes = [value.shape for value in next(iter(values.values()))]
			initial_values = [np.zeros((1, *value_shape)) for value_shape in value_shapes]

			client_count = self.simulation.client_count
			with h5py.File(path, 'w') as hf:
				for client_id in range(client_count):
					client_group = hf.create_group(str(client_id))
					for i, (initial_value, value_shape) in enumerate(zip(initial_values, value_shapes)):
						# Create a subgroup for each client, so it can hold which rounds the client participated in
						# and the values that were sent
						client_layer_group = client_group.create_group(str(i))

						client_layer_group.create_dataset(
							"server_rounds",
							data=np.array([0]),
							maxshape=(None,),
							chunks=True,
							compression="gzip",
							shuffle=True,
							fletcher32=True
						)
						max_layer_shape = (None, *value_shape)
						client_layer_group.create_dataset(
							"values",
							data=initial_value,
							maxshape=max_layer_shape,
							chunks=(1, *value_shape),
							compression="gzip",
							shuffle=True,
							fletcher32=True
						)

		with h5py.File(path, 'r+') as hf:
			for client_id in values.keys():
				client_group = hf[str(client_id)]
				for i, value in enumerate(values[client_id]):
					client_layer_group = client_group[str(i)]
					included_round_total = client_layer_group["server_rounds"].shape[0]
					client_layer_group["values"].resize((included_round_total + 1, *value.shape))
					client_layer_group["server_rounds"].resize((included_round_total + 1,))

					if isinstance(value, torch.Tensor):
						value = value.cpu().numpy()
					client_layer_group["values"][-1] = value
					client_layer_group["server_rounds"][-1] = server_round

	def _capture_aggregate_round(
			self,
			server_round: int,
			values: List[npt.NDArray],
			path: str,
			is_parameters: bool = False,
	):
		# If nothing has been captured yet, the file needs to be initialized
		# The initial parameter is all zeros for the metric and the initial model parameters for an aggregate
		if server_round == 1:
			# Create the values that are put in the hdf5 file initial row
			value_shapes = [value.shape for value in values]
			if is_parameters:
				initial_values = [np.array([layer]) for layer in training.get_weights(self.simulation.model)]
			else:
				initial_values = [np.zeros((1, *value_shape)) for value_shape in value_shapes]

			with h5py.File(path, 'w') as hf:
				for i, (initial_value, value_shape) in enumerate(zip(initial_values, value_shapes)):
					max_shape = (None, *value_shape)
					hf.create_dataset(
						str(i),
						data=initial_value,
						maxshape=max_shape,
						chunks=(1, *value_shape),
						compression="gzip",
						shuffle=True,
						fletcher32=True
					)

		with h5py.File(path, 'r+') as hf:
			for i, value in enumerate(values):
				hf[str(i)].resize((server_round + 1, *value.shape))
				hf[str(i)][server_round] = value


def get_objects_memory_usage():
	objects = gc.get_objects()  # Get a list of all objects tracked by the garbage collector

	result = []
	for obj in objects:
		try:
			size = sys.getsizeof(obj)  # Get the size of the object
			result.append((obj, size))
		except TypeError:
			# In case the object type is not supported by sys.getsizeof()
			continue

	return result
