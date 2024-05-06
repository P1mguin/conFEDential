from __future__ import annotations

import yaml

from .Attack import Attack
from .Simulation import Simulation


class Config:
	def __init__(self, simulation: Simulation, attack: Attack):
		self._simulation = simulation
		self._attack = attack

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
		pass