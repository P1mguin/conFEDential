from __future__ import annotations

from typing import List, Tuple

from torch.utils.data import DataLoader

import src.utils as utils
from src import federated_datasets


class Dataset:
	def __init__(self, name: str, splitter: dict, preprocess_fn: str) -> None:
		self.name = name
		self.alpha = float(splitter['alpha'])
		self.percent_non_iid = float(splitter['percent_non_iid'])
		self.raw_preprocess_fn = preprocess_fn
		self.preprocess_fn = utils.load_func_from_function_string(preprocess_fn, "preprocess_fn")

	def __str__(self) -> str:
		result = "Dataset"
		result += f"\n\tname: {self.name}"
		result += "\n\tpreprocess_fn:\n\t\t{}".format("\n\t\t".join(self.raw_preprocess_fn.split("\n")))
		result += f"\n\tsplitter:\n\t\talpha: {self.alpha}\n\t\tpercent_non_iid: {self.percent_non_iid}"
		return result

	def __repr__(self) -> str:
		result = "Dataset("
		result += f"name={self.name}, "
		result += f"alpha={self.alpha}, "
		result += f"percent_non_iid={self.percent_non_iid})"
		return result

	@staticmethod
	def from_dict(config: dict) -> Dataset:
		return Dataset(**config)

	def get_dataloaders(self, client_count: int, batch_size: int) -> Tuple[List[DataLoader], DataLoader]:
		dataclass = getattr(federated_datasets, self.name)
		return dataclass.load_data(client_count, batch_size, self.preprocess_fn, self.alpha, self.percent_non_iid)

	def get_name(self) -> str:
		return self.name
