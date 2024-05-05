from __future__ import annotations

import yaml

from .Attack import Attack
from .Simulation import Simulation


class Config:
	def __init__(self, simulation: Simulation, attack: Attack):
		self.simulation = simulation
		self.attack = attack

	def __str__(self):
		result = "Config:"
		result += "\n\t{}".format("\n\t".join(str(self.simulation).split("\n")))
		result += "\n\t{}".format("\n\t".join(str(self.attack).split("\n")))
		return result

	def __repr__(self):
		result = "Config("
		result += f"{repr(self.simulation)}, "
		result += f"{repr(self.attack)}"
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
