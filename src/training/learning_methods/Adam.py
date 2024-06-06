from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
from flwr.common import FitRes, Parameters
from flwr.server.client_proxy import ClientProxy
from numpy import typing as npt
from torch import nn as nn
from torch.utils.data import DataLoader

from src.training.learning_methods.Strategy import Strategy


class Adam(Strategy):
	def __init__(self, **kwargs):
		super(Adam, self).__init__(**kwargs)

	def get_optimizer(self, parameters: Iterator[nn.Parameter]) -> torch.optim.Optimizer:
		return torch.optim.Adam(parameters, **self.kwargs)

	def train(
			self,
			parameters: List[npt.NDArray],
			train_loader: DataLoader,
			simulation,
			metrics: Dict[str, Any]
	) -> Tuple[List[npt.NDArray], int, Dict[str, Any]]:
		raise NotImplementedError

	def aggregate_fit(
			self,
			server_round: int,
			results: List[Tuple[ClientProxy, FitRes]],
			failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
			simulation
	) -> Tuple[Optional[Parameters], Dict[str, Any]]:
		raise NotImplementedError
