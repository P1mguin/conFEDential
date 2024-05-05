from __future__ import annotations

from .Data import Data
from .Federation import Federation
from .Model import Model


class Simulation:
	def __init__(self, data: Data, federation: Federation, model: Model):
		self.data = data
		self.federation = federation
		self.model = model

	def __str__(self):
		result = "Simulation:"
		result += "\n\t{}".format("\n\t".join(str(self.data).split("\n")))
		result += "\n\t{}".format("\n\t".join(str(self.federation).split("\n")))
		result += "\n\t{}".format("\n\t".join(str(self.model).split("\n")))
		return result

	def __repr__(self):
		result = "Simulation("
		result += f"{repr(self.data)}, "
		result += f"{repr(self.federation)}, "
		result += f"{repr(self.model)}"
		result += ")"
		return result

	@staticmethod
	def from_dict(config: dict) -> Simulation:
		return Simulation(
			data=Data.from_dict(config['data']),
			federation=Federation.from_dict(config['federation']),
			model=Model.from_dict(config['model'])
		)
