from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
from flwr.common import FitRes, Parameters
from flwr.server.client_proxy import ClientProxy
from numpy import typing as npt
from torch import nn as nn
from torch.utils.data import DataLoader

from src.training.strategies.Strategy import Strategy
from src.utils.old_configs import Config


class RMSprop(Strategy):
	def __init__(self, **kwargs):
		super(RMSprop, self).__init__(**kwargs)

	def aggregate_fit(
			self,
			server_round: int,
			results: List[Tuple[ClientProxy, FitRes]],
			failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
			run_config: Config
	) -> Tuple[Optional[Parameters], Dict[str, Any]]:
		raise NotImplementedError

	def get_optimizer_instance(self, parameters: Iterator[nn.Parameter]) -> torch.optim.Optimizer:
		return torch.optim.RMSprop(parameters, **self.kwargs)

	def train(
			self,
			parameters:
			List[npt.NDArray],
			train_loader: DataLoader,
			run_config: Config,
			config: Dict[str, Any]
	) -> Tuple[List[npt.NDArray], int, Dict[str, Any]]:
		raise NotImplementedError
