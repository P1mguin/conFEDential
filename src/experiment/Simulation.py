import hashlib
import json
import math
import multiprocessing
import os
import pickle
from logging import ERROR, INFO, WARN
from typing import Generator, List

import flwr as fl
import h5py
import psutil
import ray
import torch
import wandb
from fedartml import SplitAsFederatedData
from flwr.common.logger import log
from torch.utils.data import DataLoader, Dataset, random_split

from src.experiment import Data, Federation, Model
from src.training import Client, Server

PROJECT_DIRECTORY = os.path.abspath(os.path.join(os.getcwd(), "./"))


class Simulation:
	def __init__(self, cache_root, data, federation, model):
		self._cache_root = cache_root
		self._data = data
		self._federation = federation
		self._model = model

		# Set the train and test loaders based on the dataset and the federation
		self._train_loaders = None
		self._test_loader = None
		self._non_member_loader = None

		log(INFO, "Preparing datasets for federated learning simulation")
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
	def from_dict(config: dict, cache_root: str) -> 'Simulation':
		return Simulation(
			cache_root=cache_root,
			data=Data.from_dict(config['data'], cache_root),
			federation=Federation.from_dict(config['federation']),
			model=Model.from_dict(config['model'], cache_root)
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
	def non_member_loader(self) -> DataLoader:
		return self._non_member_loader

	@non_member_loader.setter
	def non_member_loader(self, non_member_loader: DataLoader):
		self._non_member_loader = non_member_loader

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

		capture_directory = f"{self._cache_root}data/{dataset}/training/{model_name}/{optimizer}/{simulation_hash}/"

		# Ensure the directory exists
		os.makedirs(capture_directory, exist_ok=True)

		return capture_directory

	def get_optimizer(self, parameters):
		return self._model.get_optimizer(parameters)

	def simulate_federation(
			self,
			concurrent_clients: int,
			memory: int | None,
			num_cpus: int,
			num_gpus: int,
			is_ray_initialised: bool = False,
			is_capturing: bool = False,
			is_online: bool = False,
			run_name: str = None
	):
		"""
		Simulates federated learning for the given dataset, federation and model.
		"""
		client_fn = Client.get_client_fn(self)
		strategy = Server(self, is_capturing)

		wandb_kwargs = self.get_wandb_kwargs(run_name)
		mode = "online" if is_online else "offline"
		wandb.init(mode=mode, **wandb_kwargs)

		client_resources = get_client_resources(concurrent_clients, memory, num_cpus, num_gpus)
		log(INFO, "Starting federated learning simulation")
		try:
			if is_ray_initialised:
				ray.init(address='auto')
				fl.simulation.start_simulation(
					client_fn=client_fn,
					num_clients=self.client_count,
					client_resources=client_resources,
					keep_initialised=is_ray_initialised,
					config=fl.server.ServerConfig(num_rounds=int(1e4)),
					strategy=strategy
				)
			else:
				ray_init_args = get_ray_init_args(memory, num_cpus, num_gpus)
				fl.simulation.start_simulation(
					client_fn=client_fn,
					num_clients=self.client_count,
					client_resources=client_resources,
					ray_init_args=ray_init_args,
					config=fl.server.ServerConfig(num_rounds=int(1e4)),
					strategy=strategy
				)
			wandb.finish()
		except Exception as e:
			log(ERROR, e)
			wandb.finish(exit_code=1)

		# Shut ray down
		ray.shutdown()

	def get_server_aggregates(self, aggregate_access_indices):
		"""
		Returns the aggregates of the model parameters and metrics of the simulation over all global rounds.
		"""
		# Get the file names of the aggregates and metrics
		base_path = self.get_capture_directory()
		base_path = f"{base_path}aggregates/"
		aggregate_file = f"{base_path}aggregates.hdf5"
		metric_directory = f"{base_path}metrics/"
		metric_files = [metric_directory + file for file in os.listdir(metric_directory)]

		# Get and return the variables
		aggregates = self._get_captured_aggregates(aggregate_file, aggregate_access_indices)
		metrics = {}
		for metric_file in metric_files:
			metric_name = ".".join(metric_file.split("/")[-1].split(".")[:-1])
			metrics[metric_name] = self._get_captured_aggregates(metric_file, aggregate_access_indices)

		return aggregates, metrics

	def _get_captured_aggregates(self, path, aggregate_access_indices):
		aggregate_rounds = []
		with h5py.File(path, 'r') as hf:
			for layer_index in range(len(hf.keys())):
				value = hf[str(layer_index)][aggregate_access_indices]
				aggregate_rounds.append(value)
		return aggregate_rounds

	def get_messages(self, intercepted_client_ids):
		"""
		Returns the messages of the simulation.
		"""
		# Get the file names of the aggregates and metrics
		base_path = self.get_capture_directory()
		base_path = f"{base_path}messages/"
		messages_file = f"{base_path}messages.hdf5"
		metric_directory = f"{base_path}metrics/"
		metric_files = [metric_directory + file for file in os.listdir(metric_directory)]

		# Get the clients to which messages the attacker has access
		messages = self._get_captured_messages(messages_file, intercepted_client_ids)
		metrics = {}
		for metric_file in metric_files:
			metric_name = ".".join(metric_file.split("/")[-1].split(".")[:-1])
			metrics[metric_name] = self._get_captured_messages(metric_file, intercepted_client_ids)
		return messages, metrics

	def _get_captured_messages(self, path, intercepted_client_ids, aggregate_access_indices):
		def get_client_messages(client_id):
			with h5py.File(path, 'r') as hf:
				client_group = hf[str(client_id)]
				client_layers = []
				server_rounds = None
				for client_layer_group_key in range(len(client_group.keys())):
					if server_rounds is None:
						server_rounds = client_group[str(client_layer_group_key)]["server_rounds"][:]

					values = client_group[str(client_layer_group_key)]["values"][aggregate_access_indices]
					client_layers.append(values)
				yield server_rounds, client_layers

		captured_messages = [list(get_client_messages(id)) for id in intercepted_client_ids]
		return captured_messages

	def _prepare_loaders(self):
		"""
		If the data has been split for the given amount of clients and the given splitter return the cached file.
		Otherwise, load the preprocessed train and test data, split it (non-iid or iid), convert to data loaders, save
		and set the train and test loaders.
		:return:
		"""
		split_cache_path = self._get_split_cache_path()
		split_hash = split_cache_path.split("/")[-1].split(".")[0]

		# If the split dataloaders are available, load it
		if os.path.exists(split_cache_path):
			log(INFO, f"Found previously split dataloaders with hash {split_hash}, loading them")
			with open(split_cache_path, "rb") as file:
				self.train_loaders, self.test_loader, self.non_member_loader = pickle.load(file)
				return
		else:
			log(INFO, f"Found no previously split dataloaders with hash {split_hash}, splitting the data now")

		train_dataset, test_dataset, non_member_dataset = self._data.dataset

		# Split the train dataset
		train_datasets = self._split_data(train_dataset)

		# Convert the test dataset to a tuple
		test_dataset = [(torch.tensor(value["x"]), value["y"]) for value in test_dataset]
		non_member_dataset = [(torch.tensor(value["x"]), value["y"]) for value in non_member_dataset]

		# Bundle the information in a dataloader
		train_loaders = []
		for train_dataset in train_datasets:
			if self._data.batch_size < 0:
				batch_size = len(train_dataset) // abs(self._data.batch_size)
			else:
				batch_size = self._data.batch_size
			data_loader = DataLoader(train_dataset, batch_size=batch_size)
			train_loaders.append(data_loader)

		# Create the test loader
		test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
		non_member_loader = DataLoader(non_member_dataset, batch_size=len(non_member_dataset))

		# Cache the train and test loaders
		split_preprocessed_directory = self._data.get_preprocessed_cache_directory() + "split/"
		os.makedirs(split_preprocessed_directory, exist_ok=True)
		with open(split_cache_path, "wb") as file:
			pickle.dump((train_loaders, test_loader, non_member_loader), file)

		self.train_loaders = train_loaders
		self.test_loader = test_loader
		self.non_member_loader = non_member_loader

	def _get_split_cache_path(self):
		"""
		Returns the path to the cache file that contains the split data. The split is based on the non-iid splitter
		of the data object and on the amount of clients.
		"""
		base_path = self._data.get_preprocessed_cache_directory()

		# Get the hash of the amount of clients, and the splitter
		split_configuration = {
			"client_count": self._federation.client_count,
			"batch_size": self.batch_size,
		}

		if self._data.is_split:
			split_configuration.update(self._data.splitter)

		split_string = json.dumps(split_configuration, sort_keys=True)
		split_hash = hashlib.sha256(split_string.encode()).hexdigest()
		split_cache_path = f"{base_path}split/{split_hash}.pkl"
		return split_cache_path

	def _split_data(self, dataset) -> Generator[Dataset, None, None]:
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
			split_data = ([(torch.tensor(value["x"]), value["y"]) for value in subset] for subset in subsets)

		return split_data

	def get_wandb_kwargs(self, run_name: str = None):
		if run_name is None:
			tags = ["Training"]
		else:
			tags = ["Training", run_name]

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


def get_client_resources(
		concurrent_clients: int, memory: int | None, num_cpus: int | None, num_gpus: int | None
) -> dict:
	"""
	Finds the amount of resources available to each client based on the amount of desired concurrent clients and the
	resources available to the system.
	"""
	if num_cpus is None:
		total_cpus = multiprocessing.cpu_count()
	else:
		total_cpus = num_cpus

	if num_gpus is None:
		total_gpus = torch.cuda.device_count()
	else:
		total_gpus = num_gpus

	if memory is None:
		memory = psutil.virtual_memory().total
	else:
		memory = math.floor(memory * (1024 ** 3))

	client_cpus = total_cpus // concurrent_clients
	client_gpus = total_gpus / concurrent_clients
	client_memory = memory // concurrent_clients

	if client_cpus * concurrent_clients < total_cpus:
		log(WARN, "The amount of clients is not a divisor of the total amount of CPUs,\n"
				  "consider changing the amount of clients so that all available resources are used."
				  f"The total resources are {total_cpus} CPUs and {total_gpus} GPUs.")
	else:
		log(INFO, f"Created {concurrent_clients} clients with resources {client_cpus} CPUs, {client_gpus} GPUs, "
				  f"and {round(client_memory / (1024 ** 3), 1)}GB for the total available {total_cpus} CPUs, {total_gpus} GPUs "
				  f"and {round(memory / (1024 ** 3), 1)}GB.")

	client_resources = {
		"num_cpus": client_cpus,
		"num_gpus": client_gpus,
		"memory": client_memory,
	}
	return client_resources


def get_ray_init_args(memory: int | None, num_cpus: int | None, num_gpus: int | None) -> dict:
	"""
	Returns the ray init arguments for the type of system the simulation is run on.
	"""
	if num_cpus is None:
		num_cpus = multiprocessing.cpu_count()

	if num_gpus is None:
		num_gpus = torch.cuda.device_count()

	if memory is None:
		memory = psutil.virtual_memory().total
	else:
		memory = math.floor(memory * (1024 ** 3))

	ray_init_args = {
		"runtime_env": {
			"working_dir": PROJECT_DIRECTORY,
			"excludes": [".git", "hpc_runs"]
		},
		"num_cpus": num_cpus,
		"num_gpus": num_gpus,
		"_memory": memory,
	}

	return ray_init_args
