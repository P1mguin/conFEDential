import os.path
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple

from torch.utils.data import DataLoader


class Dataset(ABC):
	@staticmethod
	@abstractmethod
	def load_data(
			client_count: int,
			batch_size: int,
			preprocess_fn: Callable[[dict], dict],
			alpha: float = 1.,
			percent_non_iid: float = 0.,
			seed: int = 78
	) -> Tuple[List[DataLoader], DataLoader]:
		"""
		Load data is called for all datasets such that the train data loaders and test data loader is returned.
		The list of train data loaders is of size client_count and represents the data of the clients in the federation,
		the data is drawn non-iid via FedArtML. Evaluation is done centrally and so this function only returns one test
		data loader with all test data.
		:param client_count: The amount of clients in the federation
		:param batch_size: The batch size
		:param preprocess_fn: The function that is applied to each element to preprocess
		:param alpha: Concentration parameter of the Dirichlet distribution defining the desired degree of non-IID-ness
		for the labels of the federated data
		:param percent_non_iid: Percentage (between 0 and 100) desired of non-IID-ness for the labels of the federated data
		:param seed: Controls the starting point of the random number generator. Value is passed for reproducibility,
		use your favourite number.
		"""
		pass

	@staticmethod
	def is_data_downloaded(dataset: str) -> bool:
		"""
		Checks if a given dataset is downloaded
		:param dataset: name of the dataset
		:return: True if the dataset is downloaded, otherwise raise FileNotFoundError
		"""
		if not os.path.isdir(f".cache/{dataset}"):
			raise FileNotFoundError(
				f"Was unable to open cache for {dataset}, download it using the command:\npython download_dataset.py --dataset {dataset}")
		return True
