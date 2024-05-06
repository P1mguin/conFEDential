from __future__ import annotations

from .ModelArchitecture import ModelArchitecture


class AttackSimulation:
	def __init__(
			self,
			batch_size: int,
			optimizer_name: str,
			model_architecture: ModelArchitecture,
			optimizer_parameters: dict | None = None
	):
		if optimizer_parameters is None:
			optimizer_parameters = {}
		self._batch_size = batch_size
		self._optimizer_name = optimizer_name
		self._optimizer_parameters = optimizer_parameters
		self._model_architecture = model_architecture

	def __str__(self):
		result = "Attack Simulation:"
		result += f"\n\tbatch_size: {self._batch_size}"
		result += f"\n\toptimizer_name: {self._optimizer_name}"
		result += "\n\toptimizer_parameters:"
		result += "\n\t\t{}".format("\n\t\t".join([f"{key}: {value}" for key, value in self._optimizer_parameters.items()]))
		result += "\n\t{}".format("\n\t".join(str(self._model_architecture).split("\n")))
		return result

	def __repr__(self):
		result = "AttackSimulation("
		result += f"batch_size={self._batch_size}, "
		result += f"optimizer={self._optimizer_name}("
		result += "{})".format(", ".join([f"{key}={value}" for key, value in self._optimizer_parameters.items()]))
		result += ")"
		return result

	@staticmethod
	def from_dict(config: dict) -> AttackSimulation:
		return AttackSimulation(
			batch_size=config['batch_size'],
			optimizer_name=config['optimizer_name'],
			optimizer_parameters=config['optimizer_parameters'],
			model_architecture=ModelArchitecture.from_dict(config['model_architecture'])
		)
