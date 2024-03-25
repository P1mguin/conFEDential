from abc import ABC, abstractmethod
from typing import Callable, List, Tuple

from torch.utils.data import DataLoader


class Dataset(ABC):
	@staticmethod
	@abstractmethod
	def load_data(client_count: int, batch_size: int, preprocess_fn: Callable[[dict], dict], alpha: float = 1.,
				  percent_noniid: float = 0., seed: int = 78) -> Tuple[List[DataLoader], DataLoader]:
		pass
