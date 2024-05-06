from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
from flwr.common import FitRes, Parameters
from flwr.server.client_proxy import ClientProxy
from numpy import typing as npt
from torch import nn as nn
from torch.utils.data import DataLoader

from src.experiment import Simulation
from src.training.learning_methods.Strategy import Strategy


class RMSprop(Strategy):
	def __init__(self, **kwargs):
		super(RMSprop, self).__init__(**kwargs)

	def get_optimizer(self, parameters: Iterator[nn.Parameter]) -> torch.optim.Optimizer:
		return torch.optim.RMSprop(parameters, **self.kwargs)

	def train(
			self,
			parameters: List[npt.NDArray],
			train_loader: DataLoader,
			simulation: Simulation,
			metrics: Dict[str, Any]
	) -> Tuple[List[npt.NDArray], int, Dict[str, Any]]:
		raise NotImplementedError

	def aggregate_fit(
			self,
			server_round: int,
			results: List[Tuple[ClientProxy, FitRes]],
			failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
			simulation: Simulation
	) -> Tuple[Optional[Parameters], Dict[str, Any]]:
		raise NotImplementedError
