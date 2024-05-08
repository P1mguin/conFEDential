from __future__ import annotations

import collections
import hashlib
import json
import os
import pickle
from logging import ERROR
from typing import Generator, List

import flwr as fl
import numpy as np
import wandb
from fedartml import SplitAsFederatedData
from flwr.common.logger import log
from torch.utils.data import DataLoader, Dataset, random_split

from src.training import Client, Server
from .Data import Data
from .Federation import Federation
from .Model import Model
from .. import training

PROJECT_DIRECTORY = os.path.abspath(os.path.join(os.getcwd(), "./"))


class Simulation:
	def __init__(self, data: Data, federation: Federation, model: Model):
		self._data = data
		self._federation = federation
		self._model = model

		# Set the train and test loaders based on the dataset and the federation
		self.train_loaders = None
		self.test_loader = None
		self._prepare_loaders()

	def __str__(self):
		result = "Simulation:"
		result += "\n\t{}".format("\n\t".join(str(self._data).split("\n")))
		result += "\n\t{}".format("\n\t".join(str(self._federation).split("\n")))
		result += "\n\t{}".format("\n\t".join(str(self._model).split("\n")))
		return result

	def __repr__(self):
		result = "Simulation("
		result += f"{repr(self._data)}, "
		result += f"{repr(self._federation)}, "
		result += f"{repr(self._model)}"
		result += ")"
		return result

	@staticmethod
	def from_dict(config: dict) -> Simulation:
		return Simulation(
			data=Data.from_dict(config['data']),
			federation=Federation.from_dict(config['federation']),
			model=Model.from_dict(config['model'])
		)

	@property
	def train_loaders(self) -> List[DataLoader] | None:
		return self._train_loaders

	@train_loaders.setter
	def train_loaders(self, train_loaders: List[DataLoader]):
		self._train_loaders = train_loaders

	@property
	def test_loader(self) -> DataLoader:
		return self._test_loader

	@test_loader.setter
	def test_loader(self, test_loader: DataLoader):
		self._test_loader = test_loader

	@property
	def model(self):
		return self._model.model

	@property
	def criterion(self):
		return self._model.criterion

	@property
	def local_rounds(self):
		return self._federation.local_rounds

	@property
	def learning_method(self):
		return self._model.learning_method

	@property
	def client_count(self):
		return self._federation.client_count

	@property
	def fraction_fit(self):
		return self._federation.fraction_fit

	@property
	def global_rounds(self):
		return self._federation.global_rounds

	@property
	def batch_size(self):
		return self._data.batch_size

	@property
	def model_config(self):
		return self._model

	@property
	def data(self):
		return self._data

	def get_capture_directory(self) -> str:
		dataset = self._data.dataset_name
		model_name = self._model.model_name
		optimizer = self._model.optimizer_name

		# Get hash of the simulation
		simulation_string = str(self)
		simulation_hash = hashlib.sha256(simulation_string.encode()).hexdigest()

		capture_directory = f".cache/data/{dataset}/training/{model_name}/{optimizer}/{simulation_hash}/"

		# Ensure the directory exists
		os.makedirs(capture_directory, exist_ok=True)

		return capture_directory

	def get_optimizer(self, parameters):
		return self._model.get_optimizer(parameters)

	def simulate_federation(
			self,
			client_resources: dict,
			is_capturing: bool = False,
			is_online: bool = False,
			run_name: str = None
	):
		"""
		Simulates federated learning for the given dataset, federation and model.
		"""
		client_fn = Client.Client.get_client_fn(self)
		strategy = Server.Server(self, is_capturing)

		wandb_kwargs = self._get_wandb_kwargs(run_name)
		mode = "online" if is_online else "offline"
		wandb.init(mode=mode, **wandb_kwargs)

		ray_init_args = get_ray_init_args()

		try:
			fl.simulation.start_simulation(
				client_fn=client_fn,
				num_clients=self.client_count,
				client_resources=client_resources,
				config=fl.server.ServerConfig(num_rounds=self._federation.global_rounds),
				ray_init_args=ray_init_args,
				strategy=strategy
			)
		except Exception as e:
			log(ERROR, e)
			wandb.finish(exit_code=1)

	def get_server_aggregates(self):
		"""
		Returns the server traffic of the simulation.
		"""
		capture_directory = self.get_capture_directory()
		aggregate_directory = f"{capture_directory}aggregates/"
		aggregate_file_path = f"{aggregate_directory}aggregates.npz"

		aggregates = []
		with open(aggregate_file_path, "rb") as f:
			saved_aggregates = np.load(f)

			for file in saved_aggregates.files:
				aggregates.append(saved_aggregates[file])

		# Prepend the initial model parameters to the aggregates
		model = self.model
		initial_parameters = training.get_weights(model)
		aggregates = [np.insert(aggregates[i], 0, initial_parameters[i], axis=0) for i in range(len(aggregates))]

		metric_directory = f"{aggregate_directory}metrics/"
		metrics = collections.defaultdict(list)
		for metric_file in os.listdir(metric_directory):
			metric_name = ".".join(metric_file.split(".")[:-1])
			with open(metric_directory + metric_file, "rb") as f:
				metric = np.load(f)

				metric_values = []
				for file in metric.files:
					metric_values.append(metric[file])

				# Prepend zero like parameters to the metric aggregates
				metric_values = [
					np.insert(metric_values[i], 0, np.zeros_like(metric_values[i][0]), axis=0)
					for i in range(len(metric_values))
				]
				metrics[metric_name] = metric_values

		return aggregates, dict(metrics)


	def _prepare_loaders(self):
		"""
		If the data has been split for the given amount of clients and the given splitter return the cached file.
		Otherwise, load the preprocessed train and test data, split it (non-iid or iid), convert to data loaders, save
		and set the train and test loaders.
		:return:
		"""
		split_cache_path = self._get_split_cache_path()

		# If the split dataloaders are available, load it
		if os.path.exists(split_cache_path):
			with open(split_cache_path, "rb") as file:
				self.train_loaders, self.test_loader = pickle.load(file)
				return

		train_dataset, test_dataset = self._data.dataset

		# Split the train dataset
		train_datasets = self._split_data(train_dataset)

		# Convert the test dataset to a tuple
		test_dataset = [(value["x"], value["y"]) for value in test_dataset]

		# Bundle the information in a dataloader
		train_loaders = []
		for train_dataset in train_datasets:
			if self._data.batch_size == -1:
				batch_size = len(train_dataset)
			else:
				batch_size = self._data.batch_size
			data_loader = DataLoader(train_dataset, batch_size=batch_size)
			train_loaders.append(data_loader)

		# Create the test loader
		test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

		# Cache the train and test loaders
		split_preprocessed_directory = self._data.get_preprocessed_cache_directory() + "split/"
		os.makedirs(split_preprocessed_directory, exist_ok=True)
		with open(split_cache_path, "wb") as file:
			pickle.dump((train_loaders, test_loader), file)

		self.train_loaders = train_loaders
		self.test_loader = test_loader

	def _get_split_cache_path(self):
		"""
		Returns the path to the cache file that contains the split data. The split is based on the non-iid splitter
		of the data object and on the amount of clients.
		"""
		base_path = self._data.get_preprocessed_cache_directory()

		# Get the hash of the amount of clients, and the splitter
		split_configuration = {
			"client_count": self._federation.client_count
		}

		if self._data.is_split:
			split_configuration.update(self._data.splitter)

		split_string = json.dumps(split_configuration, sort_keys=True)
		split_hash = hashlib.sha256(split_string.encode()).hexdigest()
		split_cache_path = f"{base_path}split/{split_hash}.pkl"
		return split_cache_path

	def _split_data(self, dataset: Dataset) -> Generator[Dataset]:
		# Split the data based on the non-iid splitter
		client_count = self._federation.client_count
		if self._data.is_split:
			splitter = self._data.splitter
			split_data, _, _, _ = SplitAsFederatedData(random_state=78).create_clients(
				image_list=dataset["x"],
				label_list=dataset["y"],
				num_clients=client_count,
				method="dirichlet",
				alpha=splitter["alpha"],
				percent_noniid=splitter["percent_non_iid"]
			)

			# Use with class completion so every client has at least one label of each class
			split_data = split_data["with_class_completion"].values()
		else:
			lengths = [dataset.shape[0] // client_count] * client_count
			remainder = dataset.shape[0] % client_count

			# Add 1 to the first `remainder` clients
			for i in range(remainder):
				lengths[i] += 1

			subsets = random_split(dataset, lengths)
			split_data = ([(value["x"], value["y"]) for value in subset] for subset in subsets)

		return split_data

	def _get_wandb_kwargs(self, run_name: str = None):
		if run_name is None:
			tags = []
		else:
			tags = [run_name]

		return {
			"project": "conFEDential",
			"tags": tags,
			"config": {
				"dataset": self._data.dataset_name,
				"model": self._model.model_name,
				"learning_method": self._model.optimizer_name,
				"batch_size": self._data.batch_size,
				"client_count": self._federation.client_count,
				"fraction_fit": self._federation.fraction_fit,
				"local_rounds": self._federation.local_rounds,
				**self._model.optimizer_parameters
			}
		}


def get_ray_init_args():
	ray_init_args = {
		"runtime_env": {
			"working_dir": PROJECT_DIRECTORY,
			"excludes": [".git", "hpc_runs"]
		},
	}

	# Cluster admin wants to use local instead of tmp
	if os.path.exists("/local"):
		ray_init_args = {
			**ray_init_args,
			"_temp_dir": "/local/ray",
		}

	return ray_init_args
