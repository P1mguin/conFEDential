from __future__ import annotations

from typing import List, Type

import flwr.common
import torch
import torch.nn as nn

from src.training import helper


class Model:
	def __init__(self, name: str, criterion: str, layers: List[dict]) -> None:
		self.name = name
		self.criterion = getattr(nn, criterion)

		model_layers = []
		for layer in layers:
			layer_type = getattr(nn, layer["type"])
			parameters = {key: value for key, value in list(layer.items())[1:]}
			layer = layer_type(**parameters)
			model_layers.append(layer)

		class Net(nn.Module):
			def __init__(self) -> None:
				super(Net, self).__init__()
				self.layers = nn.Sequential(*model_layers)

			def forward(self, x: torch.Tensor) -> torch.Tensor:
				x = self.layers(x)
				return x

		self.model = Net

	def __str__(self) -> str:
		result = "Model"
		result += f"\n\tname: {self.name}"
		result += f"\n\tcriterion: {str(self.get_criterion_instance())}"
		result += f"\n\tlayers:"
		for line in str(self.get_model_instance()).split("\n"):
			result += f"\n\t\t{line}"
		return result

	def __repr__(self) -> str:
		result = "Model("
		result += f"name={self.name}, "
		result += f"criterion={str(self.get_criterion_instance())}"
		return result

	@staticmethod
	def from_dict(config: dict) -> Model:
		return Model(**config)

	def get_criterion_class(self) -> Type[nn.Module]:
		return self.criterion

	def get_criterion_instance(self) -> nn.Module:
		return self.criterion()

	def get_initial_parameters(self):
		return flwr.common.ndarrays_to_parameters(helper.get_weights_from_model(self.get_model_instance()))

	def get_model_class(self) -> Type[nn.Module]:
		return self.model

	def get_model_instance(self) -> nn.Module:
		return self.model()

	def get_name(self) -> str:
		return self.name
