from __future__ import annotations

from typing import List


class Model:
	def __init__(
			self,
			optimizer_name: str,
			model_name: str,
			criterion_name: str,
			optimizer_parameters: dict,
			model_architecture: List[dict]
	):
		self.optimizer_name = optimizer_name
		self.model_name = model_name
		self.criterion_name = criterion_name
		self.optimizer_parameters = optimizer_parameters
		self.model_architecture = model_architecture

	def __str__(self):
		result = "Model:"
		result += f"\n\toptimizer_name: {self.optimizer_name}"
		result += f"\n\tmodel_name: {self.model_name}"
		result += f"\n\tcriterion_name: {self.criterion_name}"
		result += "\n\toptimizer_parameters: \n\t\t{}".format("\n\t\t".join([f"{key}: {value}" for key, value in self.optimizer_parameters.items()]))
		result += "\n\tmodel_architecture:"
		for layer in self.model_architecture:
			result += "\n\t\t{}".format("\n\t\t\t".join([f"{key}: {value}" for key, value in layer.items()]))
		return result

	def __repr__(self):
		result = "Model("
		result += f"optimizer={self.optimizer_name}("
		result += "{}), ".format(", ".join([f"{key}={value}" for key, value in self.optimizer_parameters.items()]))
		result += f"model_name={self.model_name}, "
		result += f"criterion_name={self.criterion_name}"
		result += ")"
		return result

	@staticmethod
	def from_dict(config: dict) -> Model:
		return Model(
			optimizer_name=config['optimizer_name'],
			model_name=config['model_name'],
			criterion_name=config['criterion_name'],
			optimizer_parameters=config['optimizer_parameters'],
			model_architecture=config['model_architecture']
		)
