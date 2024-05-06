from __future__ import annotations

import hashlib
from typing import List, Tuple

from torch.utils.data import DataLoader

import src.utils as utils
from src import datasets


class Dataset:
	"""
	A class that represents the dataset on which the experiment will be performed.
	name: the name of the dataset loaded from the Hugging Face library. Before calling get_dataloaders, this dataset
	should be downloaded in .cache
	splitter: an optional dictionary that should contain alpha and percent_non_idd. These values are used to dirichlet
	split the data using FedArtML.
	preprocess_fn: a function that is applied to each element before it is returned, represented as a string.
	"""

	def __init__(self, name: str, preprocess_fn: str, splitter: dict | None=None) -> None:
		self.name = name
		self.has_splitter = splitter is not None
		if self.has_splitter:
			self.alpha = float(splitter['alpha'])
			self.percent_non_iid = float(splitter['percent_non_iid'])
		else:
			self.alpha = None
			self.percent_non_iid = None
		self.raw_preprocess_fn = preprocess_fn
		self.preprocess_fn = utils.load_func_from_function_string(preprocess_fn, "preprocess_fn")

	def __str__(self) -> str:
		result = "Dataset"
		result += f"\n\tname: {self.name}"
		result += "\n\tpreprocess_fn:\n\t\t{}".format("\n\t\t".join(self.raw_preprocess_fn.split("\n")))
		if self.has_splitter:
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
		"""
		Loads in a dataset instance from a dictionary
		:param config: the configuration dictionary
		"""
		return Dataset(**config)

	def get_dataloaders(self, client_count: int, batch_size: int) -> Tuple[List[DataLoader], DataLoader]:
		dataclass = getattr(datasets, self.name)

		raw_preprocess_fn = self.get_raw_preprocess_fn()
		hash_object = hashlib.sha256(raw_preprocess_fn.encode())
		function_hash = hash_object.hexdigest()

		return dataclass.load_data(
			client_count=client_count,
			batch_size=batch_size,
			preprocess_fn=self.preprocess_fn,
			alpha=self.alpha,
			percent_non_iid=self.percent_non_iid,
			function_hash=function_hash
		)

	def get_name(self) -> str:
		return self.name

	def get_raw_preprocess_fn(self):
		return self.raw_preprocess_fn
