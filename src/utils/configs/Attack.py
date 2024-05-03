import random
from enum import Enum
from typing import Iterator, List

import numpy as np
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader

from src.utils.configs.AttackConfig import AttackConfig
from src.utils.configs.AttackModel import AttackModel
from src.utils.configs.Simulation import Simulation


class DataAccessType(Enum):
	CLIENT = 1
	ALL = 2


class UpdateAccessType(Enum):
	SERVER = 1
	SERVER_ENCRYPTED = 2
	CLIENT = 3


class Attack:
	"""
	A class that represents the access the attacker has during the simulation, can be either one of three options:
	- server: The attacker has access to all the client updates and all model aggregates
	- server-encrypted: The attacker has access to no client updates and all model aggregates
	- client: The attacker has access to the client updates and model aggregates that were generated and received by
	some random client
	"""

	def __init__(
			self,
			data_access_type: str,
			update_access_type: str,
			shadow_model_amount: int,
			targeted_attack: bool,
			attack_model: AttackModel,
			simulation: Simulation
	):
		self.data_access_type = DataAccessType[data_access_type.upper()]
		self.update_access_type = UpdateAccessType[update_access_type.upper()]
		self.shadow_model_amount = shadow_model_amount
		self.targeted_attack = targeted_attack
		self.attack_model = attack_model
		self.simulation = simulation

		# In the learning process there is always at least two clients required by the Flower Framework. Therefore,
		# we can set client 0 as the default attacker
		if self.update_access_type == UpdateAccessType.CLIENT or self.data_access_type == DataAccessType.CLIENT:
			self.attacker_id = 0
		else:
			self.attacker_id = None

		# Determine whether the item is in the dataset
		if self.targeted_attack:
			self.target_member = None
			self.is_member = bool(random.getrandbits(1))

	def __str__(self):
		result = "Attack"
		result += f"\n\tdata_access_type: {self.data_access_type}"
		result += f"\n\tupdate_access_type: {self.update_access_type}"
		result += f"\n\ttarget_attack: {self.targeted_attack}"
		result += f"\n\tshadow_model_amount: {self.get_shadow_model_amount()}"
		result += "\n\t{}".format('\n\t'.join(str(self.simulation).split('\n')))
		result += "\n\t{}".format('\n\t'.join(str(self.attack_model).split('\n')))
		return result

	def __repr__(self):
		result = "Attack("
		result += f"data_access_type={self.data_access_type}, "
		result += f"update_access_type={self.update_access_type}, "
		result += f"shadow_model_amount={self.get_shadow_model_amount()}"
		result += f"{repr(self.simulation)}, "
		result += ")"
		return result

	@staticmethod
	def from_dict(config: dict):
		attack_model = AttackModel.from_dict(config.pop("attack_model"))
		simulation = Simulation.from_dict(config.pop("simulation"))
		return Attack(attack_model=attack_model, simulation=simulation, **config)

	def get_attack_batch_size(self) -> int:
		return self.simulation.get_batch_size()

	def get_attack_model(self, run_config: AttackConfig):
		return self.attack_model.get_attack_model(run_config)

	def get_attack_optimizer(self, parameters: Iterator[nn.Parameter]) -> torch.optim.Optimizer:
		return self.simulation.get_optimizer_instance(parameters)

	def get_attack_optimizer_name(self) -> str:
		return self.simulation.get_optimizer_name()

	def get_attacker_participation_rounds(self, capture_output_directory: str) -> List[int]:
		# Open the parameters file
		parameters_file = np.load(f"{capture_output_directory}parameters.npz")

		# Get the first layer of the parameters
		client_layer = parameters_file[parameters_file.files[0]][self.attacker_id]

		# Get the rounds in which the attacker participated
		client_participation_indices = [i for i, update in enumerate(client_layer) if np.any(update)]

		return client_participation_indices

	def get_target_member(self, train_loaders: List[DataLoader], test_loader: DataLoader) -> tuple:
		if self.target_member is not None:
			return self.target_member

		if self.is_member:
			# Get the first item from the second client, since we might assume the first to be the attacker
			target_member = train_loaders[1].dataset[0]
		else:
			# Get the first item from the test loader
			target_member = test_loader.dataset[0]
			target_member = (target_member["x"], target_member["y"])

		self.target_member = target_member
		return self.target_member

	def get_is_targeted_attack(self) -> bool:
		return self.targeted_attack

	def get_is_member(self) -> bool:
		return self.is_member

	def get_model_aggregate_indices(self, global_rounds: int, capture_output_directory: str = "") -> List[int]:
		if (self.update_access_type == UpdateAccessType.SERVER
				or self.update_access_type == UpdateAccessType.SERVER_ENCRYPTED):
			return list(range(-1, global_rounds, 1))

		# Load in the rounds to which the attacker participated
		attacker_participation_indices = self.get_attacker_participation_rounds(capture_output_directory)

		# The attacker received the model prior to that round, decrement each idnex
		attacker_participation_indices = [i - 1 for i in attacker_participation_indices]
		return attacker_participation_indices

	def get_client_update_indices(self, client_count: int) -> List[int]:
		if self.update_access_type == UpdateAccessType.SERVER:
			return list(range(client_count))
		elif self.update_access_type == UpdateAccessType.SERVER_ENCRYPTED:
			return []
		else:
			return [self.attacker_id]

	def get_attack_data_indices(self, client_count: int) -> List[int]:
		if self.data_access_type == DataAccessType.ALL:
			return list(range(client_count))
		elif self.data_access_type == DataAccessType.CLIENT:
			return [self.attacker_id]
		else:
			return []

	def get_shadow_model_amount(self):
		return self.shadow_model_amount

	def get_data_access_type(self) -> DataAccessType:
		return self.data_access_type

	def get_update_access_type(self) -> UpdateAccessType:
		return self.update_access_type
