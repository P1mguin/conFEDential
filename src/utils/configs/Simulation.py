from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, Tuple, Type

import torch
import torch.nn as nn


class Optimizer:
	def __init__(self, optimizer_name: str, **kwargs) -> None:
		self.optimizer_name = optimizer_name
		self.kwargs = kwargs

	def __repr__(self) -> str:
		result = f"Optimizer(optimizer_name={self.optimizer_name}, "
		result += ', '.join([f"{key}={value}" for key, value in self.kwargs.items()])
		result += ")"
		return result

	def __str__(self) -> str:
		result = f"Optimizer: {self.optimizer_name}"
		for key, value in self.kwargs.items():
			result += f"\n\t{key}: {value}"
		return result

	@staticmethod
	def from_dict(config: dict) -> Optimizer:
		return Optimizer(**config)

	def get_kwargs(self) -> Dict[str, Any]:
		return self.kwargs

	def get_name(self):
		return self.optimizer_name

	def get_optimizer_instance(self, parameters: Iterator[nn.Parameter]) -> Type[torch.optim.Optimizer]:
		return getattr(torch.optim, self.optimizer_name)(parameters, **self.kwargs)

	def get_optimizer_spawner(self) -> Callable[[Iterator[nn.Parameter]], Type[torch.optim.Optimizer]]:
		return lambda parameters: getattr(torch.optim, self.optimizer_name)(parameters, **self.kwargs)


class Simulation:
	def __init__(
			self,
			batch_size: int,
			client_count: int,
			fraction_fit: float,
			global_rounds: int,
			local_rounds: int,
			optimizer: Optimizer
	) -> None:
		self.batch_size = batch_size
		self.client_count = client_count
		self.fraction_fit = fraction_fit
		self.global_rounds = global_rounds
		self.local_rounds = local_rounds
		self.optimizer = optimizer

	def __repr__(self) -> str:
		result = "Simulation("
		for key, value in list(self.__dict__.items())[:-1]:
			result += f"{key}={value}, "
		result += f"{repr(self.optimizer)})"
		return result

	def __str__(self) -> str:
		result = "Simulation"
		for key, value in list(self.__dict__.items())[:-1]:
			result += f"\n\t{key}: {value}"
		result += "\n\t{}".format('\n\t'.join(str(self.optimizer).split('\n')))
		return result

	@staticmethod
	def from_dict(config: dict) -> Simulation:
		optimizer = Optimizer.from_dict(config.pop("optimizer"))
		return Simulation(optimizer=optimizer, **config)

	def get_batch_size(self) -> int:
		return self.batch_size

	def get_client_count(self) -> int:
		return self.client_count

	def get_client_selection_config(self) -> Tuple[float, float, int, int, int]:
		return (
			self.get_fraction_evaluate(),
			self.get_fraction_fit(),
			self.get_min_available_clients(),
			self.get_min_evaluate_clients(),
			self.get_min_fit_clients(),
		)

	def get_fraction_evaluate(self) -> float:
		return 0.

	def get_fraction_fit(self) -> float:
		return self.fraction_fit

	def get_global_rounds(self) -> int:
		return self.global_rounds

	def get_local_rounds(self) -> int:
		return self.local_rounds

	def get_min_available_clients(self) -> int:
		return max(self.get_min_evaluate_clients(), self.get_min_fit_clients())

	def get_min_evaluate_clients(self) -> int:
		return int(self.get_client_count() * self.get_fraction_evaluate())

	def get_min_fit_clients(self) -> int:
		return max(int(self.get_client_count() * self.get_fraction_fit()), 1)

	def get_optimizer_instance(self, parameters: Iterator[nn.Parameter]) -> Type[torch.optim.Optimizer]:
		return self.optimizer.get_optimizer_instance(parameters)

	def get_optimizer_kwargs(self) -> Dict[str, Any]:
		return self.optimizer.get_kwargs()

	def get_optimizer_name(self):
		return self.optimizer.get_name()
