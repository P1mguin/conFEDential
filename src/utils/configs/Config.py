from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterator, List, Tuple, Type

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

import src.utils.configs as configs

project_name = "conFEDential"


class Config:
	def __init__(self, simulation, dataset, model) -> None:
		self.simulation = simulation
		self.dataset = dataset
		self.model = model

	def __repr__(self) -> str:
		return "Config({})".format(", ".join([f"{repr(value)}" for key, value in self.__dict__.items()]))

	def __str__(self) -> str:
		result = "Config"
		for key, value in self.__dict__.items():
			result += "\n\t{}".format('\n\t'.join(str(value).split('\n')))
		return result

	@staticmethod
	def from_yaml_file(file_path: str) -> Config:
		with open(file_path, "r") as f:
			yaml_file = yaml.safe_load(f)

		return Config.from_dict(yaml_file)

	@staticmethod
	def from_dict(config: dict) -> Config:
		kwargs = {key: getattr(configs, key.capitalize()).from_dict(value) for key, value in config.items()}
		return Config(**kwargs)

	def get_batch_size(self) -> int:
		return self.simulation.get_batch_size()

	def get_client_count(self) -> int:
		return self.simulation.get_client_count()

	def get_client_selection_config(self) -> Tuple[float, float, int, int, int]:
		return self.simulation.get_client_selection_config()

	def get_criterion(self) -> nn.Module:
		return self.model.get_criterion_instance()

	def get_dataloaders(self) -> Tuple[List[DataLoader], DataLoader]:
		client_count = self.get_client_count()
		batch_size = self.get_batch_size()
		return self.dataset.get_dataloaders(client_count=client_count, batch_size=batch_size)

	def get_dataset_name(self) -> str:
		return self.dataset.get_name()

	def get_global_rounds(self) -> int:
		return self.simulation.get_global_rounds()

	def get_initial_parameters(self):
		return self.model.get_initial_parameters()

	def get_local_rounds(self) -> int:
		return self.simulation.get_local_rounds()

	def get_model(self) -> nn.Module:
		return self.model.get_model_instance()

	def get_model_name(self) -> str:
		return self.model.get_name()

	def get_optimizer(self, parameters: Iterator[nn.Parameter]) -> Type[torch.optim.Optimizer]:
		return self.simulation.get_optimizer_instance(parameters)

	def get_optimizer_name(self) -> str:
		return self.simulation.get_optimizer_name()

	def get_output_capture_file_path(self) -> str:
		dataset = self.get_dataset_name()
		model = self.get_model_name()
		optimizer = self.simulation.get_optimizer_name()
		time = datetime.now().strftime("%Y-%m-%d_%H-%M")
		path = f".captured/{dataset}/{model}/{optimizer}/{time}.npz"
		return path

	def get_wandb_kwargs(self, batch_name: str = None) -> Dict[str, Any]:
		if batch_name is None:
			tags = []
		else:
			tags = [batch_name]

		return {
			"project": project_name,
			"tags": tags,
			"config": {
				"dataset": self.get_dataset_name(),
				"model": self.get_model_name(),
				"batch_size": self.get_batch_size(),
				"client_count": self.get_client_count(),
				"fraction_fit": self.get_client_selection_config()[1],
				"local_rounds": self.get_local_rounds(),
				**self.simulation.get_optimizer_kwargs()
			}
		}
