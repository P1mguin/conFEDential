from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import pickle
import random
from logging import INFO
from typing import List, Tuple

import numpy as np
import torch.utils.data
import yaml
from flwr.common import log
from torch.utils.data import DataLoader

from .Attack import Attack
from .Simulation import Simulation
from .. import training


class Config:
	def __init__(self, simulation: Simulation, attack: Attack):
		self._simulation = simulation
		self._attack = attack

		if self._attack.is_targeted and self._attack.data_access == "client":
			# Set the client id we are simulating the attack from
			self.attack.client_id = random.randint(0, self.simulation.client_count - 1)

	def __str__(self):
		result = "Config:"
		result += "\n\t{}".format("\n\t".join(str(self._simulation).split("\n")))
		result += "\n\t{}".format("\n\t".join(str(self._attack).split("\n")))
		return result

	def __repr__(self):
		result = "Config("
		result += f"{repr(self._simulation)}, "
		result += f"{repr(self._attack)}"
		result += ")"
		return result

	@staticmethod
	def from_yaml_file(yaml_file: str) -> Config:
		with open(yaml_file, "r") as f:
			config = yaml.safe_load(f)

		return Config.from_dict(config)

	@staticmethod
	def from_dict(config: dict) -> Config:
		return Config(
			simulation=Simulation.from_dict(config['simulation']),
			attack=Attack.from_dict(config['attack'])
		)

	@property
	def simulation(self) -> Simulation:
		return self._simulation

	@property
	def attack(self) -> Attack:
		return self._attack

	def run_simulation(self, client_resources: dict, is_online: bool, is_capturing: bool, run_name: str):
		# If nothing has been captured for this simulation, capture the simulation
		federation_simulation_capture_directory = self.simulation.get_capture_directory()
		if not os.path.exists(federation_simulation_capture_directory):
			self.simulation.simulate_federation(client_resources, is_online, is_capturing, run_name)

		# Get the shadow models
		target_shadow_models = self._get_shadow_models()
		pass

	def _get_shadow_models(self):
		"""
		Gets the shadow models used to attack the system. It returns a list with as many items as there are targets,
		in case the attack is untargeted it is a list of size 1. Each item in the list is a tuple; the first item
		is again a tuple that contains: the target, whether they are a member another tuple: the dataset used to train
		the corresponding shadow model, and whether the target was a member in that dataset. The second item in the
		tuple is the several iterations of the shadow model: the parameters and the metrics.

		If the attack is untargeted, the first item of the tuple will contain a list of the tuples of the shadow model
		datasets. The first item is a member, and the second not.
		"""
		shadow_model_file = self._get_shadow_model_cache_path()

		# If the shadow models have already been trained return them
		if os.path.exists(shadow_model_file):
			with open(shadow_model_file, "rb") as f:
				return pickle.load(f)

		if self.attack.is_targeted:
			targets = self._get_targeted_shadow_model_datasets()
		else:
			targets = self._get_untargeted_shadow_model_datasets()

		target_shadow_models = []
		for i, target in enumerate(targets):
			if self.attack.is_targeted:
				log(INFO, f"Training shadow models for target: {i}")
				shadow_model_datasets = [shadow_model_dataset[0] for shadow_model_dataset in target[2]]
			else:
				shadow_model_datasets = [shadow_model_dataset for shadow_model_dataset in target[1]]

			shadow_models = []
			for j, shadow_model_dataset in enumerate(shadow_model_datasets):
				log(INFO, f"Training shadow model: {j}")
				# Train a shadow model
				parameters, metrics = self._train_shadow_model(shadow_model_dataset)
				shadow_models.append((parameters, metrics))
			target_shadow_models.append(shadow_models)

		target_shadow_models = list(zip(targets, target_shadow_models))

		# Save the shadow models
		os.makedirs(os.path.dirname(shadow_model_file), exist_ok=True)
		with open(shadow_model_file, "wb") as f:
			pickle.dump(target_shadow_models, f)
		return target_shadow_models

	def _get_shadow_model_cache_path(self) -> str:
		base_path = self.simulation.get_capture_directory()

		target = "targeted" if self.attack.is_targeted else "untargeted"

		# Get the hash of the attack config except the attack simulation and the message access
		attack_config = {
			"is_targeted": self.attack.is_targeted,
			"data_access": self.attack.data_access,
			"shadow_model": self.attack.shadow_model_amount,
			"targets": self.attack.targets,
		}
		attack_string = json.dumps(attack_config, sort_keys=True)
		attack_hash = hashlib.sha256(attack_string.encode()).hexdigest()

		attack_model_path = f"{base_path}shadowing/{attack_hash}/{target}/models.pkl"
		return attack_model_path

	def _train_shadow_model(self, dataloader):
		"""
		Trains a shadow model given a dataloader and returns the parameters of each iteration and the metrics
		"""
		# Train in an identical way to the federation
		strategy = self.simulation.learning_method
		global_rounds = self.simulation.global_rounds

		# Set the initial state
		model = self.simulation.model
		parameters = training.get_weights(model)
		metrics = {}

		# Capture the intermediate states
		parameter_iterations = [parameters]
		metric_iterations = [metrics]
		for i in range(global_rounds):
			parameters, _, metrics = strategy.train(parameters, dataloader, self.simulation, metrics)
			parameter_iterations.append(parameters)
			metric_iterations.append(metrics)

		# Reshape the captured parameters and metrics so that for each variable it is a list where each
		# index correspond to the ith iteration, i.e. transpose it
		parameters = list(map(list, zip(*parameter_iterations)))

		# Set the initial metric value to a null shaped matrix
		for key, value in metric_iterations[1].items():
			null_value = []
			for layer in value:
				null_value.append(np.zeros_like(layer))
			metric_iterations[0][key] = null_value

		# Transpose the metrics
		metrics = collections.defaultdict(list)
		for i, metric_iteration in enumerate(metric_iterations):
			for key, value in metric_iteration.items():
				metrics[key].append(value)
		for key, value in metrics.items():
			metrics[key] = list(map(list, zip(*value)))

		return parameters, dict(metrics)

	def _get_targeted_shadow_model_datasets(self):
		"""
		Function that generates the datasets on which the shadow models will be trained. It generates a list of targets
		for each of which a shadow dataset is created. A shadow dataset is generated by combining all the data to
		which the attacker has access, and removing the target from it. Then the target is added to half of the dataset.
		It returns a list of tuples, each containing the target, whether they are a member of the target model and
		the shadow model dataset
		"""

		# Get the targets
		targets = self._get_targets()

		# Get the data to which the attacker has access
		data_access_indices = self.attack.get_data_access_indices(self.simulation.client_count)
		train_loaders = self.simulation.train_loaders
		train_loaders = [train_loader for i, train_loader in enumerate(train_loaders) if i in data_access_indices]

		# Get the shadow model datasets for each target
		shadow_model_datasets = []
		for target, is_member in targets:
			# Merge the data loaders in one big dataset
			dataset = (sample for train_loader in train_loaders for sample in train_loader.dataset)

			# Create a copy of the dataset in which the target is not present
			dataset_without_target = [sample for sample in dataset if sample is not target]

			# Scale the batch size so it is relatively equal
			client_batch_size = self.simulation.batch_size
			batch_size = client_batch_size * int(len(dataset_without_target) / self.simulation.client_count)
			batch_size = pow(2, round(math.log2(abs(batch_size))))

			target_dataset = []
			for i in range(self.attack.shadow_model_amount):
				# For the first half create a dataloader without the target, the other half has it
				in_first_half = i < self.attack.shadow_model_amount // 2
				if in_first_half:
					data_loader = DataLoader(dataset_without_target, batch_size=batch_size, shuffle=True)
				else:
					data_loader = DataLoader(
						[*dataset_without_target, target], batch_size=batch_size, shuffle=True
					)
				target_dataset.append((data_loader, in_first_half))
			shadow_model_datasets.append((target, is_member, target_dataset))
		return shadow_model_datasets

	def _get_untargeted_shadow_model_datasets(self):
		"""
		Function that generates the datasets on which the shadow models will be trained. For each shadow model,
		it splits the data to which the attacker has access in half. The first half will be used in a shadow model,
		and the other half will be excluded from training to better model the relation for membership. It returns
		a list of tuples, each containing a random half split of the complete dataset
		"""
		# Get the data to which the attacker has access
		data_access_indices = self.attack.get_data_access_indices(self.simulation.client_count)
		train_loaders = self.simulation.train_loaders
		train_loaders = [train_loader for i, train_loader in enumerate(train_loaders) if i in data_access_indices]

		# Merge the data loaders in one big dataset
		dataset = [sample for train_loader in train_loaders for sample in train_loader.dataset]
		total_length = len(dataset)
		member_length = total_length // 2
		non_member_length = total_length - member_length

		# Scale the batch size so it is relatively equal
		client_batch_size = self.simulation.batch_size
		batch_size = client_batch_size * int(len(dataset) / 2 / self.simulation.client_count)
		batch_size = pow(2, round(math.log2(abs(batch_size))))

		shadow_model_datasets = []
		for i in range(self.attack.shadow_model_amount):
			member_dataset, non_member_dataset = torch.utils.data.random_split(dataset,
																			   [member_length, non_member_length])
			member_dataset = DataLoader(member_dataset, batch_size=batch_size, shuffle=True)
			non_member_dataset = DataLoader(non_member_dataset, batch_size=batch_size, shuffle=True)
			shadow_model_datasets.append((member_dataset, non_member_dataset))

		return [shadow_model_datasets]

	def _get_targets(self) -> List[Tuple[tuple, bool]]:
		train_loaders = self.simulation.train_loaders
		test_loader = self.simulation.test_loader

		# Get the client indices from which the target can be chosen
		possible_targets = list(range(self.simulation.client_count))
		if self.attack.data_access == "client":
			possible_targets.remove(self.attack.client_id)

		targets = []
		for _ in range(self.attack.targets):
			# Determine if the target is a member
			is_member = bool(random.getrandbits(1))
			if is_member:
				# Choose a random index from the possible targets
				target_client = random.choice(possible_targets)

				# Choose a random element from the target client's dataset
				target_index = random.randint(0, len(train_loaders[target_client].dataset) - 1)
				target = train_loaders[target_client].dataset[target_index]
			else:
				# Choose a random element from the test loader dataset
				target_index = random.randint(0, len(test_loader.dataset) - 1)
				target = test_loader.dataset[target_index]
			targets.append((target, is_member))

		return targets
