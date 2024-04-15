from __future__ import annotations

from typing import List

import yaml

import src.utils.configs as configs
from src.utils.configs import Attack, Dataset, Model, Simulation
from src.utils.configs.Config import Config


class AttackConfig(Config):
	"""
	An attack configuration is a configuration that is used to define the attack simulation. It is a subclass of the
	normal configuration and contains additional information on the attack strategy and the attack target.
	"""

	def __init__(self, attack: Attack, simulation: Simulation, dataset: Dataset, model: Model) -> None:
		super().__init__(simulation, dataset, model)
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

	def get_target_member(self) -> int:
		return self.attack.get_target_member()

	def get_model_aggregate_indices(self, client_count: int) -> List[int]:
		return self.attack.get_model_aggregate_indices(client_count)

	def get_client_update_indices(self, client_count: int) -> List[int]:
		return self.attack.get_client_update_indices(client_count)

	def is_target_member(self) -> bool:
		return self.attack.is_target_member()

	def set_target_member(self) -> None:
		# Get the dataloaders
		train_loaders, test_loader = self.get_dataloaders()

		# Set the target member at the attack object
		self.attack.set_target_member(train_loaders, test_loader)


if __name__ == '__main__':
	yaml_file_path = "examples/confidentiality_simulation/single_experiments/mnist/logistic_regression/fed_avg.yaml"
	attack_config = AttackConfig.from_yaml_file(yaml_file_path)
