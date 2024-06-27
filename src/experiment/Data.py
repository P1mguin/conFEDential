import hashlib
import os
import pickle
from logging import INFO

from flwr.common import log

from src import data


class Data:
	def __init__(self, cache_root: str, dataset_name: str, batch_size: int, raw_preprocess_fn: str | None = None,
				 splitter: dict | None = None) -> None:
		# Set the attributes
		self._cache_root = cache_root
		self._dataset_name = dataset_name.lower()
		self._batch_size = batch_size
		self._raw_preprocess_fn = raw_preprocess_fn
		self._is_split = splitter is not None
		if self._is_split:
			self._alpha = float(splitter['alpha'])
			self._percent_non_iid = float(splitter['percent_non_iid'])

		# Load the preprocess function
		if raw_preprocess_fn is not None:
			namespace = {}
			exec(self._raw_preprocess_fn, namespace)
			self._preprocess_fn = namespace['preprocess_fn']
		else:
			self._raw_preprocess_fn = ""
			self._preprocess_fn = lambda x: x

		# Prepare the train and test data
		self._dataset = None

		log(INFO, "Preprocessing the federated learning dataset")
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
	def from_dict(config: dict, cache_root) -> 'Data':
		return Data(
			cache_root=cache_root,
			dataset_name=config['dataset_name'],
			batch_size=config['batch_size'],
			raw_preprocess_fn=config.get('preprocess_fn'),
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
		preprocessed_hash = preprocessed_cache_path.split("/")[-2]

		# If the preprocessed dataset is available, load it
		if os.path.exists(preprocessed_cache_path):
			log(INFO,
				f"Found preprocessed data for the given preprocess function with hash {preprocessed_hash}, returning")
			with open(preprocessed_cache_path, "rb") as file:
				self.dataset = pickle.load(file)
				return
		else:
			log(INFO,
				f"No preprocessed data found for the given preprocess function with hash {preprocessed_hash}, preprocessing now")

		# Load the raw dataset
		train_dataset, test_dataset, non_member_dataset = getattr(data, self._dataset_name).load_dataset(
			self._cache_root)

		# Apply the preprocess function to the train and test dataset
		train_dataset = train_dataset.map(self._preprocess_fn)
		test_dataset = test_dataset.map(self._preprocess_fn)
		non_member_dataset = non_member_dataset.map(self._preprocess_fn)

		# Save the preprocessed dataset
		with open(preprocessed_cache_path, "wb") as file:
			pickle.dump((train_dataset, test_dataset, non_member_dataset), file)

		self.dataset = (train_dataset, test_dataset, non_member_dataset)

	def get_preprocessed_cache_directory(self):
		"""
		Returns the path to the cache directory for the dataset with the given preprocess function.
		"""
		base_path = f"{self._cache_root}data/{self._dataset_name}/preprocessed/"

		# Get the hash function of the preprocess function
		function_hash = hashlib.sha256(self._raw_preprocess_fn.encode()).hexdigest()
		preprocessed_cache_directory = f"{base_path}{function_hash}/"

		# Create the directory if it does not exist yet
		os.makedirs(preprocessed_cache_directory, exist_ok=True)

		return preprocessed_cache_directory

	def _get_unsplit_preprocessed_cache_file(self):
		"""
		Returns the path to the cache file for the unsplit preprocessed dataset.
		"""
		preprocessed_cache_directory = self.get_preprocessed_cache_directory()

		# Return the path
		preprocessed_cache_path = f"{preprocessed_cache_directory}unsplit.pkl"
		return preprocessed_cache_path
