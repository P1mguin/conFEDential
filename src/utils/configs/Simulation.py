from __future__ import annotations

from typing import Any, Dict, Iterator, Tuple, Type

import torch
import torch.nn as nn

import src.training.strategies as strategies


class Optimizer:
	"""
	A class that represents the optimizer the experiment will run with.
	optimizer_name: The name of an optimizer function of torch.optim that will be used to compute the update to the
	model parameters
	kwargs: All key word arguments except the model parameters that are required to summon the optimizer
	"""

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
		"""
		Returns the optimizer object from a dictionary
		:param config: The configuration dictionary
		"""
		return Optimizer(**config)

	def get_strategy(self) -> strategies.Strategy:
		return getattr(strategies, self.optimizer_name)(**self.get_kwargs())

	def get_name(self):
		return self.optimizer_name

	def get_kwargs(self) -> Dict[str, Any]:
		return self.kwargs

	def get_optimizer_instance(self, parameters: Iterator[nn.Parameter]) -> torch.optim.Optimizer:
		return self.get_strategy().get_optimizer_instance(parameters)


class Simulation:
	"""
	A class that represents the simulation configuration of the experiment.
	batch_size: The batch size that the clients will train with
	client_count: The number of clients that the simulation will run with
	fraction_fit: The fraction of clients that are selected each round, i.e. if client_count = 100, fraction_fit = 0.1,
	the server will train with 10 clients in each round
	global_rounds: The amount of times the server will ask a fraction of clients to train on their data
	local_rounds: The amount of times the client will repeat the learning process, i.e. epochs
	optimizer: A valid configuration for an Optimizer instance
	"""

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
		# In the current code base, the server only does the evaluation. Therefore, the evaluation fraction is 0.
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

	def get_optimizer_instance(self, parameters: Iterator[nn.Parameter]) -> torch.optim.Optimizer:
		return self.optimizer.get_optimizer_instance(parameters)

	def get_optimizer_kwargs(self) -> Dict[str, Any]:
		return self.optimizer.get_kwargs()

	def get_optimizer_name(self):
		return self.optimizer.get_name()

	def get_strategy(self):
		return self.optimizer.get_strategy()
