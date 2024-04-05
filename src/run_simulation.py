import os
import sys

import wandb

from src.utils.configs import Config

PROJECT_DIRECTORY = os.path.abspath(os.path.join(os.getcwd(), "./"))
sys.path.append(PROJECT_DIRECTORY)

import argparse
from pathlib import Path
from typing import Callable, Dict, List, Tuple
from logging import INFO

from flwr.common.logger import log

import flwr as fl
import numpy.typing as npt
import torch
from flwr.common import Scalar
from torch.utils.data import DataLoader

import src.server_aggregation_strategies as agg
from src.training import helper
import random
import numpy as np

os.environ["HF_DATASETS_OFFLINE"] = "1"
torch.manual_seed(78)
random.seed(78)
np.random.seed(78)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description="Running conFEDential simulation")

parser.add_argument(
	"--yaml-file",
	type=str,
	help="Path to the yaml file that contains the configuration of the simulation"
)

parser.add_argument(
	"--num-cpus",
	type=int,
	default=144,
	help="Number of CPUs to assign to a virtual client"
)

parser.add_argument(
	"--num-gpus",
	type=float,
	default=8.,
	help="Number of GPUs to assign to a virtual client"
)


def get_client_fn(train_loaders: List[DataLoader], run_config: Config) -> Callable[[str], fl.client.Client]:
	"""
	Returns a function that can be called to summon a client that will run the simulation on the given train loader
	with the given run configuration
	:param train_loaders: the train dataset of the client
	:param run_config: the run configuration
	"""
	class FlowerClient(fl.client.NumPyClient):
		# Client does not get evaluation method, that is done at server level over all data at once
		def __init__(self, cid: int) -> None:
			self.parameters = None
			self.cid = cid

		def get_parameters(self, config: Dict[str, Scalar]) -> List[npt.NDArray]:
			return self.parameters

		def set_parameters(self, parameters: List[npt.NDArray]) -> None:
			self.parameters = parameters

		def fit(self, parameters: List[npt.NDArray], config: Dict[str, Scalar]) -> Tuple[List[npt.NDArray], int, dict]:
			self.set_parameters(parameters)
			new_parameters, data_size = helper.train(parameters, train_loaders[self.cid], run_config)
			return new_parameters, data_size, {}

	def client_fn(cid: str) -> fl.client.Client:
		cid = int(cid)
		return FlowerClient(cid).to_client()

	return client_fn


def get_evaluate_fn(
		test_loader: DataLoader,
		run_config: Config
) -> Callable[[int, List[npt.NDArray], Dict[str, Scalar]], Tuple[float, Dict[str, Scalar]]]:
	"""
	The evaluation function that the server will use to measure loss and accuracy each round. The results are tracked
	in W&B
	:param test_loader: the test data the server will test with
	:param run_config: the run configuration with which rules the server will test
	"""
	def evaluate(
			server_round: int,
			parameters: fl.common.NDArrays,
			config: Dict[str, Scalar]
	) -> Tuple[float, Dict[str, Scalar]]:
		loss, accuracy, data_size = helper.test(parameters, test_loader, run_config)
		wandb.log({"loss": loss, "accuracy": accuracy})
		return loss, {"accuracy": accuracy, "data_size": data_size}

	return evaluate


def run_simulation(config: Config, client_resources: dict, batch_run_name: str = None) -> None:
	"""
	Runs a federated learning simulation with the rules defined in the config. The results are tracked using W&B.
	:param config: the experiment configuration
	:param client_resources: the resource eac client gets
	:param batch_run_name: an identifier that will be added to the tags to mark it as being part of a batch run
	"""
	log(INFO, f"\nRunning with {config}")

	train_loaders, test_loader = config.get_dataloaders()
	client_fn = get_client_fn(train_loaders, config)
	evaluate = get_evaluate_fn(test_loader, config)

	strategy = agg.get_capturing_strategy(
		strategy=fl.server.strategy.FedAvg,
		config=config,
		evaluate_fn=evaluate
	)

	wandb_kwargs = config.get_wandb_kwargs()
	mode = "online" if batch_run_name is None else "offline"
	wandb.init(mode=mode, **wandb_kwargs)

	ray_init_args = {
		"runtime_env": {
			"working_dir": PROJECT_DIRECTORY,
		}
	}

	try:
		fl.simulation.start_simulation(
			client_fn=client_fn,
			num_clients=config.get_client_count(),
			client_resources=client_resources,
			config=fl.server.ServerConfig(num_rounds=config.get_global_rounds()),
			ray_init_args=ray_init_args,
			strategy=strategy,
		)
	except Exception as _:
		wandb.finish(exit_code=1)

	wandb.finish()


def main() -> None:
	args = parser.parse_args()

	client_resources = {
		"num_cpus": args.num_cpus,
		"num_gpus": args.num_gpus
	}

	config = Config.from_yaml_file(str(Path(args.yaml_file).resolve()))
	run_simulation(config, client_resources)



if __name__ == '__main__':
	main()
