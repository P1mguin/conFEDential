from typing import Any, Dict, List

import flwr as fl
import numpy.typing as npt
from torch.utils.data import DataLoader

from src.experiment import Simulation


class Client(fl.client.NumPyClient):
	def __init__(self, cid: int, simulation: Simulation, train_loader: DataLoader):
		self.cid = cid
		self.simulation = simulation
		self.learning_method = simulation.learning_method
		self.train_loader = train_loader

	def fit(self, parameters: List[npt.NDArray], metrics: Dict[str, Any]):
		new_parameters, data_size, metrics = self.learning_method.train(
			parameters,
			self.train_loader,
			self.simulation,
			metrics
		)
		return new_parameters, data_size, metrics

	@staticmethod
	def get_client_fn(simulation: Simulation):
		train_loaders = simulation.train_loaders

		def client_fn(cid: str) -> fl.client.Client:
			cid = int(cid)
			return Client(cid, simulation, train_loaders[cid]).to_client()

		return client_fn
