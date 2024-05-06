from __future__ import annotations

import hashlib
import os
import pickle

from src import data


class Data:
	def __init__(self, dataset_name: str, batch_size: int, raw_preprocess_fn: str,
				 splitter: dict | None = None) -> None:
		# Set the attributes
		self._dataset_name = dataset_name.lower()
		self._batch_size = batch_size
		self._raw_preprocess_fn = raw_preprocess_fn
		self._is_split = splitter is not None
		if self._is_split:
			self._alpha = float(splitter['alpha'])
			self._percent_non_iid = float(splitter['percent_non_iid'])

		# Load the preprocess function
		namespace = {}
		exec(self._raw_preprocess_fn, namespace)
		self._preprocess_fn = namespace['preprocess_fn']

		# Prepare the train and test data
		self._dataset = None
		self._prepare_dataset()

	def __str__(self):
		result = "Data:"
		result += f"\n\tdataset_name: {self._dataset_name}"
		result += f"\n\tbatch_size: {self._batch_size}"
		result += "\n\tpreprocess_fn:\n\t\t{}".format("\n\t\t".join(self._raw_preprocess_fn.split("\n")[:-1]))
		if self.is_split:
			result += "\n\tsplitter:"
			result += f"\n\t\talpha: {self._alpha}"
			result += f"\n\t\tpercent_non_iid: {self._percent_non_iid}"
		return result

	def __repr__(self):
		result = "Data("
		result += f"dataset_name={self._dataset_name}, "
		result += f"batch_size={self._batch_size}"
		if self.is_split:
			result += f", alpha={self._alpha}, "
			result += f"percent_non_iid={self._percent_non_iid}"
		result += ")"
		return result

	@staticmethod
	def from_dict(config: dict) -> Data:
		return Data(
			dataset_name=config['dataset_name'],
			batch_size=config['batch_size'],
			raw_preprocess_fn=config['preprocess_fn'],
			splitter=config.get('splitter', None)
		)

	@property
	def dataset(self):
		return self._dataset

	@dataset.setter
	def dataset(self, value):
		self._dataset = value

	@property
	def dataset_name(self):
		return self._dataset_name

	@property
	def is_split(self):
		return self._is_split

	@property
	def splitter(self):
		if self.is_split:
			return {
				"alpha": self._alpha,
				"percent_non_iid": self._percent_non_iid
			}
		else:
			raise AttributeError("This experiment is not split")

	@property
	def batch_size(self):
		return self._batch_size

	def _prepare_dataset(self):
		"""
		If the preprocessed dataset is available, loads that and sets the dataset variable. Otherwise,
		loads the raw dataset, preprocesses it, saves it, and sets the dataset variable.
		"""
		# Get the path to the preprocessed cache
		preprocessed_cache_path = self._get_unsplit_preprocessed_cache_file()

		# If the preprocessed dataset is available, load it
		if os.path.exists(preprocessed_cache_path):
			with open(preprocessed_cache_path, "rb") as file:
				self.dataset = pickle.load(file)
				return

		# Load the raw dataset
		train_dataset, test_dataset = getattr(data, self._dataset_name).load_dataset()

		# Apply the preprocess function to the train and test dataset
		train_dataset = train_dataset.map(self._preprocess_fn)
		test_dataset = test_dataset.map(self._preprocess_fn)

		# Create the cache directory if it does not exist
		cache_directory = "/".join(preprocessed_cache_path.split("/")[:-1])
		os.makedirs(cache_directory, exist_ok=True)

		# Save the preprocessed dataset
		with open(preprocessed_cache_path, "wb") as file:
			pickle.dump((train_dataset, test_dataset), file)

		self.dataset = (train_dataset, test_dataset)

	def get_preprocessed_cache_directory(self):
		"""
		Returns the path to the cache directory for the dataset with the given preprocess function.
		"""
		base_path = f".cache/data/{self._dataset_name}/preprocessed/"

		# Get the hash function of the preprocess function
		function_hash = hashlib.sha256(self._raw_preprocess_fn.encode()).hexdigest()
		preprocessed_cache_directory = f"{base_path}{function_hash}/"
		return preprocessed_cache_directory

	def _get_unsplit_preprocessed_cache_file(self):
		"""
		Returns the path to the cache file for the unsplit preprocessed dataset.
		"""
		preprocessed_cache_directory = self.get_preprocessed_cache_directory()

		# Return the path
		preprocessed_cache_path = f"{preprocessed_cache_directory}/unsplit.pkl"
		return preprocessed_cache_path
