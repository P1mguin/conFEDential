import os.path
from abc import ABC, abstractmethod
from typing import Tuple

from torch.utils.data import DataLoader


class Dataset(ABC):
	@staticmethod
	@abstractmethod
	def load_dataset() -> Tuple[DataLoader, DataLoader]:
		"""
		Load raw dataset from cache directory, if not found, throws an error
		"""
		pass

	@staticmethod
	def is_data_downloaded(dataset: str) -> bool:
		"""
		Checks if a given dataset is downloaded, otherwise throws an error
		"""
		if not os.path.isdir(f".cache/data/{dataset}"):
			raise FileNotFoundError(
				f"Was unable to open cache for {dataset}, download it using the command:\npython download_dataset.py --dataset {dataset}")
		return True