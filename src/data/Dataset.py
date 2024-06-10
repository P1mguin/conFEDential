import os.path
from abc import ABC, abstractmethod
from typing import Tuple

from datasets import Dataset as HuggingFaceDataset


class Dataset(ABC):
	@staticmethod
	@abstractmethod
	def load_dataset(cache_root) -> Tuple[HuggingFaceDataset, HuggingFaceDataset]:
		"""
		Load raw dataset from cache directory, if not found, throws an error
		"""
		pass

	@staticmethod
	def is_data_downloaded(dataset: str, cache_root) -> bool:
		"""
		Checks if a given dataset is downloaded, otherwise throws an error
		"""
		if not os.path.isdir(f"{cache_root}data/{dataset}"):
			raise FileNotFoundError(
				f"Was unable to open cache for {dataset}, download it using the command:\npython download_dataset.py --dataset {dataset}")
		return True
