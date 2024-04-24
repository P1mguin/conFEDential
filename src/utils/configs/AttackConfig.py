from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import yaml
from flwr.common import parameters_to_ndarrays
from torch.utils.data import ConcatDataset, DataLoader

import src.utils.configs as configs
from src.utils import k_fold_dataset
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

	def get_attack_data_loaders(
			self,
			train_loaders: Optional[List[DataLoader]] = None
	) -> List[Tuple[DataLoader, DataLoader]]:
		"""
		Creates a list of train and test loaders that can be used to train as many shadow models as the list is long
		:param train_loaders: the train loaders from which the new train and test loaders should be created
		"""
		# Ideally the train loaders do not have to be re-fetched since some operations are not cached
		if train_loaders is None:
			train_loaders, _ = self.get_dataloaders()

		# Get the train_loaders to which the attacker has access
		data_indices = self.get_attack_data_indices()
		train_loaders = np.array(train_loaders)
		train_loaders = train_loaders[data_indices]

		# Combine the train_loaders into one dataset
		datasets = [dataloader.dataset for dataloader in train_loaders]
		combined_dataset = ConcatDataset(datasets)

		# K-fold the combined dataset into multiple train and test loaders
		k = self.get_shadow_model_amount()
		batch_size = self.get_batch_size()
		k_folds = k_fold_dataset(combined_dataset, k, batch_size)
		return list(k_folds)

	def get_attack_data_indices(self) -> List[int]:
		client_count = self.get_client_count()
		return self.attack.get_attack_data_indices(client_count)

	def get_attack_model(self):
		return self.attack.get_attack_model(self)

	def get_shadow_model_amount(self):
		return self.attack.get_shadow_model_amount()

	def get_target_member(self) -> int:
		return self.attack.get_target_member()

	def get_model_aggregate_indices(self, capture_output_directory: str = "") -> List[int]:
		return self.attack.get_model_aggregate_indices(self.get_client_count(), capture_output_directory)

	def get_model_aggregates(self) -> Dict[str, List[npt.NDArray]]:
		output_directory = self.get_output_capture_directory_path()
		aggregate_directory = f"{output_directory}aggregates/"
		metric_directory = f"{aggregate_directory}metrics/"

		# Get the files in which the aggregates reside
		parameter_file = f"{aggregate_directory}parameters.npz"
		metric_files = [f"{metric_directory}{file}" for file in os.listdir(metric_directory)]
		file_names = [parameter_file, *metric_files]

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
		metric_files = [f"{metric_directory}{file}" for file in os.listdir(metric_directory)]
		file_names = [parameter_file, *metric_files]

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
