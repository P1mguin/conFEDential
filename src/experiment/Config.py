from __future__ import annotations

import os
import random

import numpy as np
import yaml

from .Attack import Attack
from .Simulation import Simulation


class Config:
	def __init__(self, simulation: Simulation, attack: Attack):
		self._simulation = simulation
		self._attack = attack

		if self._attack.data_access == "client":
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
		if not os.path.exists(f"{federation_simulation_capture_directory}aggregates"):
			self.simulation.simulate_federation(client_resources, is_capturing, is_online, run_name)

		# For each intercepted datapoint, get their gradients, activation functions, loss
		attack_dataset = self._get_attack_dataset()

	def _get_attack_dataset(self):
		# Get the model
		server_aggregates = self.simulation.get_server_aggregates()

		# Get the data to which the attacker has access
		attack_data = self._get_intercepted_samples()
		attack_dataset = self.attack.get_attack_dataset(server_aggregates, attack_data, self.simulation)
		return attack_dataset

	def _get_intercepted_samples(self):
		# Get the data loaders to which the attacker has access
		client_count = self.simulation.client_count
		data_access_indices = self.attack.get_data_access_indices(client_count)

		# Get the train_loaders with the corresponding indices
		all_train_loaders = self.simulation.train_loaders
		test_loader = self.simulation.test_loader
		data_loaders = np.array(all_train_loaders)[data_access_indices]

		# Get the target sample
		_, target = self.attack.get_target_sample(self.simulation)

		# If the attacker has all-access add the test data as well
		if self.attack.data_access == "all":
			# The attack would be trivial if the attacker also had access to the sample. So remove that
			training_samples = [
				(*item, True) for dataloader in data_loaders for item in dataloader.dataset if item is not target
			]
			test_samples = [
				(*item, False) for item in test_loader.dataset if item is not target
			]
		# Otherwise, assume the client has a similar ratio of test data as they have train data
		else:
			training_samples = [(*item, True) for dataloader in data_loaders for item in dataloader.dataset]

			test_data_size = int(len(test_loader.dataset) / client_count)
			start = test_data_size * self.attack.client_id
			end = test_data_size * (self.attack.client_id + 1)
			test_samples = [(*item, False) for item in test_loader.dataset[start:end]]

		intercepted_samples = training_samples + test_samples

		return intercepted_samples
