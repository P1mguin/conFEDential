from __future__ import annotations

import math
import os
import re
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch.nn as nn
import torch.optim
import yaml
from flwr.common import parameters_to_ndarrays
from torch.utils.data import ConcatDataset, DataLoader

import src.utils.configs as configs
from src.utils import split_dataloader
from src.utils.configs import Attack, Dataset, Model, Simulation
from src.utils.configs.Config import Config


class AttackConfig(Config):
	"""
	An attack configuration is a configuration that is used to define the attack simulation. It is a subclass of the
	normal configuration and contains additional information on the attack strategy and the attack target.
	"""

	def __init__(self, attack: Attack, simulation: Simulation, dataset: Dataset, model: Model) -> None:
		super(AttackConfig, self).__init__(simulation, dataset, model)
		self.attack = attack
		self.set_target_member()

	def __repr__(self) -> str:
		return "AttackConfig({})".format(", ".join([f"{repr(value)}" for key, value in self.__dict__.items()]))

	def __str__(self) -> str:
		result = "AttackConfig"
		for key, value in self.__dict__.items():
			result += "\n\t{}".format('\n\t'.join(str(value).split('\n')))
		return result

	@staticmethod
	def from_yaml_file(file_path: str) -> AttackConfig:
		"""
		Returns an AttackConfig instance from a YAML file path
		:param file_path: the path to the YAML file
		"""
		with open(file_path, "r") as f:
			yaml_file = yaml.safe_load(f)

		return AttackConfig.from_dict(yaml_file)

	@staticmethod
	def from_dict(config: dict) -> AttackConfig:
		"""
		Returns an AttackConfig instance from a dictionary with a valid structure. Does not check if the structure is valid,
		the code will likely throw an error if the dictionary is invalid.
		:param config: the configuration dictionary
		"""
		kwargs = {key: getattr(configs, key.capitalize()).from_dict(value) for key, value in config.items()}
		return AttackConfig(**kwargs)

	def get_attack_batch_size(self) -> int:
		return self.attack.get_attack_batch_size()

	def get_attack_data_loaders(
			self,
			all_train_loaders: Optional[List[DataLoader]] = None
	) -> List[Tuple[DataLoader, DataLoader]]:
		"""
		Creates a list of train and test loaders that can be used to train as many shadow models as the list is long
		:param all_train_loaders: the train loaders from which the new train and test loaders should be created
		"""
		if all_train_loaders is None:
			all_train_loaders, _ = self.get_dataloaders()

		# Get the train_loaders to which the attacker has access
		data_indices = self.get_attack_data_indices()
		all_train_loaders = np.array(all_train_loaders)
		train_loaders = all_train_loaders[data_indices]

		# Combine the train_loaders into one dataloader
		datasets = [dataloader.dataset for dataloader in train_loaders]
		combined_dataset = ConcatDataset(datasets)

		# Scale the batch size to be relatively equal size to the proportion of the client
		combined_length = len(combined_dataset)
		average_client_length = combined_length / len(all_train_loaders)

		# Scale and round to nearest power of 2
		client_batch_size = self.get_batch_size()
		batch_size = client_batch_size * int(combined_length / 2 / average_client_length)
		batch_size = pow(2, round(math.log2(abs(batch_size))))

		combined_dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

		# TODO: Split based on whether the attack is online or offline
		n = self.get_shadow_model_amount()
		data_loaders = []
		for _ in range(n):
			train_loader, test_loader = split_dataloader(combined_dataloader, 0.5)
			data_loaders.append((train_loader, test_loader))
		return data_loaders

	def get_attack_data_indices(self) -> List[int]:
		client_count = self.get_client_count()
		return self.attack.get_attack_data_indices(client_count)

	def get_attack_model(self):
		return self.attack.get_attack_model(self)

	def get_attack_optimizer(self, parameters: Iterator[nn.Parameter]) -> torch.optim.Optimizer:
		return self.attack.get_attack_optimizer(parameters)

	def get_shadow_model_amount(self):
		return self.attack.get_shadow_model_amount()

	def get_target_member(self) -> int:
		return self.attack.get_target_member()

	def get_model_aggregate_indices(self, capture_output_directory: str = "") -> List[int]:
		return self.attack.get_model_aggregate_indices(self.get_global_rounds(), capture_output_directory)

	def get_model_aggregates(self) -> Dict[str, List[npt.NDArray]]:
		output_directory = self.get_output_capture_directory_path()
		aggregate_directory = f"{output_directory}aggregates/"
		metric_directory = f"{aggregate_directory}metrics/"

		# Get the files in which the aggregates reside
		parameter_file = f"{aggregate_directory}parameters.npz"
		if os.path.exists(metric_directory):
			metric_files = [f"{metric_directory}{file}" for file in os.listdir(metric_directory)]
			file_names = [parameter_file, *metric_files]
		else:
			file_names = [parameter_file]

		# Get the iterations to which the attacker has access
		# and shift them such that the initial parameters can be taken into account
		iterations_access = self.get_model_aggregate_indices(output_directory)
		iterations_access = [i + 1 for i in iterations_access]

		# Collect the results
		model_aggregates = {}
		for file_name in file_names:
			file = np.load(file_name)
			match = re.search(r'/([^/]+)\.npz$', file_name)
			key = match.group(1)

			np_arrays = []
			for i, np_file in enumerate(file.files):
				layer = file[np_file]

				# All parameters instead of the model parameters are initialized at zero
				if key == "parameters":
					initial_value = parameters_to_ndarrays(self.get_initial_parameters())
				else:
					initial_value = np.zeros_like(layer[0])

				# Shift the layer by one such that the initial parameters can be taken into account
				shifted_layer = np.zeros((layer.shape[0] + 1, *layer.shape[1:]))
				shifted_layer[1:11] = layer
				shifted_layer[0] = initial_value[i]

				np_arrays.append(shifted_layer[iterations_access])

			model_aggregates[key] = np_arrays
		return model_aggregates

	def get_client_update_indices(self, client_count: int) -> List[int]:
		return self.attack.get_client_update_indices(client_count)

	def get_client_updates(self) -> Dict[str, List[npt.NDArray]]:
		output_directory = self.get_output_capture_directory_path()
		metric_directory = f"{output_directory}metrics/"

		# Get the files in which the client updates reside
		parameter_file = f"{output_directory}parameters.npz"
		if os.path.exists(metric_directory):
			metric_files = [f"{metric_directory}{file}" for file in os.listdir(metric_directory)]
			file_names = [parameter_file, *metric_files]
		else:
			file_names = [parameter_file]

		# Get the iterations to which the attacker has access
		iterations_access = self.get_client_update_indices(self.get_client_count())

		client_updates = {}
		for file_name in file_names:
			file = np.load(file_name)
			match = re.search(r'/([^/]+)\.npz$', file_name)
			key = match.group(1)

			np_arrays = []
			for i, np_file in enumerate(file.files):
				layer = file[np_file]
				np_arrays.append(layer[iterations_access])

			client_updates[key] = np_arrays
		return client_updates


	def is_target_member(self) -> bool:
		return self.attack.is_target_member()

	def set_target_member(self) -> None:
		# Get the dataloaders
		train_loaders, test_loader = self.get_dataloaders()

		# Set the target member at the attack object
		self.attack.set_target_member(train_loaders, test_loader)
