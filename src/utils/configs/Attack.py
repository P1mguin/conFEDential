import random
from enum import Enum
from typing import List

from torch.utils.data import DataLoader


class AccessType(Enum):
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

	def __init__(self, access_type: AccessType):
		self.access_type = access_type
		self.target_member = None

		# In the learning process there is always at least two clients required by the Flower Framework. Therefore,
		# we can set client 0 as the default attacker
		if access_type == AccessType.CLIENT:
			self.attacker_id = 0
		else:
			self.attacker_id = None

		# Determine whether the item is in the dataset
		self.is_member = bool(random.getrandbits(1))

	def __str__(self):
		result = "Attack"
		result += f"\n\taccess_type: {self.access_type}"
		return result

	def __repr__(self):
		return f"Attack(access_type={self.access_type})"

	@staticmethod
	def from_dict(config: dict):
		return Attack(**config)

	def get_target_member(self):
		return self.target_member

	def get_model_aggregate_indices(self, client_count: int) -> List[int]:
		if self.access_type == AccessType.SERVER or self.access_type == AccessType.SERVER_ENCRYPTED:
			return list(range(client_count))
		else:
			return [self.attacker_id]

	def get_client_update_indices(self, client_count: int) -> List[int]:
		if self.access_type == AccessType.SERVER:
			return list(range(client_count))
		elif self.access_type == AccessType.SERVER_ENCRYPTED:
			return []
		else:
			return [self.attacker_id]

	def is_target_member(self):
		return self.is_member

	def set_target_member(self, train_loaders: List[DataLoader], test_loader: DataLoader):
		if self.is_member:
			# Get the first item from the second client, since we might assume the first to be the attacker
			target_member = train_loaders[1].dataset[0][0]
		else:
			# Get the first item from the test loader
			target_member = test_loader.dataset[0]["x"]

		self.target_member = target_member
