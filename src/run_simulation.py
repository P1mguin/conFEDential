import os
import sys

PROJECT_DIRECTORY = os.path.abspath(os.path.join(os.getcwd(), "./"))
sys.path.append(PROJECT_DIRECTORY)

import argparse
import math
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Tuple, Type

import flwr as fl
import numpy.typing as npt
import torch
import torch.nn as nn
import wandb
from flwr.common import Scalar
from torch.utils.data import DataLoader

import src.server_aggregation_strategies as agg
from src.training import helper
import src.utils as utils

wandb.login()

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


def get_client_fn(
		train_loaders: List[DataLoader],
		epochs: int,
		model_class: Type[nn.Module],
		criterion_class: Type[nn.Module],
		optimizer_class: Callable[[Iterator[nn.Parameter]], Type[torch.optim.Optimizer]]
) -> Callable[[str], fl.client.Client]:
	"""
	Returns a client representing a training party in the federation
	:param train_loaders: the list of data loaders where each index $i$ represents the data of client $i$
	:param epochs: the amount of local rounds each client will run
	:param model_class: the model class that the clients will train, initialization takes no parameters
	:param criterion_class: the criterion class that the clients will train for, initialization takes no parameters
	:param optimizer_class: the optimizer class that the clients will train with, initialization only takes the
	parameters of a model
	"""

	class FlowerClient(fl.client.NumPyClient):
		# Client does not get evaluation method, that is done at server level over all data at once
		def __init__(self, cid: int):
			self.parameters = None
			self.cid = cid

		def get_parameters(self, config: dict) -> List[npt.NDArray]:
			return self.parameters

		def set_parameters(self, parameters: List[npt.NDArray]) -> None:
			self.parameters = parameters

		def fit(self, parameters: List[npt.NDArray], config: dict) -> Tuple[List[npt.NDArray], int, dict]:
			self.set_parameters(parameters)

			new_parameters, data_size = helper.train(
				epochs,
				parameters,
				model_class,
				train_loaders[self.cid],
				criterion_class,
				optimizer_class
			)

			return new_parameters, data_size, {}

	def client_fn(cid: str) -> fl.client.Client:
		cid = int(cid)
		return FlowerClient(cid).to_client()

	return client_fn


def get_evaluate_fn(
		test_loader: DataLoader,
		model_class: Type[nn.Module],
		criterion_class: Type[nn.Module]
) -> Callable[[int, fl.common.NDArrays, Dict[str, Scalar]], Tuple[float, Dict[str, Scalar]]]:
	def evaluate(
			server_round: int,
			parameters: fl.common.NDArrays,
			config: Dict[str, Scalar]
	) -> Tuple[float, Dict[str, Scalar]]:
		loss, accuracy, data_size = helper.test(parameters, model_class, test_loader, criterion_class)

		wandb.log({"loss": loss, "accuracy": accuracy})

		return loss, {"accuracy": accuracy, "data_size": data_size}

	return evaluate


def main() -> None:
	args = parser.parse_args()

	client_resources = {
		"num_cpus": args.num_cpus,
		"num_gpus": args.num_gpus
	}

	yaml_file = str(Path(args.yaml_file).resolve())
	config = utils.load_yaml_file(yaml_file)

	model_class = utils.load_model_from_yaml_file(yaml_file)
	criterion_class = getattr(torch.nn, config["model"]["criterion"])
	optimizer_class = utils.load_optimizer_from_yaml_file(yaml_file)
	train_loaders, test_loader = utils.load_data_loaders_from_config(config)
	epochs = config["simulation"]["local_rounds"]
	global_rounds = config["simulation"]["global_rounds"]

	client_fn = get_client_fn(train_loaders, epochs, model_class, criterion_class, optimizer_class)

	client_count = config["simulation"]["client_count"]

	fraction_fit = config["simulation"]["fraction_fit"]
	fraction_evaluate = 0.
	min_fit_clients = max(math.floor(fraction_fit * client_count), 1)
	min_evaluate_clients = 0
	evaluate = get_evaluate_fn(test_loader, model_class, criterion_class)
	initial_parameters = fl.common.ndarrays_to_parameters(helper.get_weights_from_model(model_class()))

	capture_name = utils.get_capture_path_from_config(config)

	strategy = agg.get_capturing_class(
		strategy=fl.server.strategy.FedAvg,
		client_count=client_count,
		output_path=capture_name,
		fraction_fit=fraction_fit,
		fraction_evaluate=fraction_evaluate,
		min_fit_clients=min_fit_clients,
		min_evaluate_clients=min_evaluate_clients,
		min_available_clients=min_fit_clients,
		evaluate_fn=evaluate,
		initial_parameters=initial_parameters
	)

	wandb_kwargs = utils.get_wandb_kwargs(config)
	wandb.init(**wandb_kwargs)
	ray_init_args = {
		"runtime_env": {
			"working_dir": PROJECT_DIRECTORY,
		}
	}

	try:
		fl.simulation.start_simulation(
			client_fn=client_fn,
			num_clients=client_count,
			client_resources=client_resources,
			config=fl.server.ServerConfig(num_rounds=global_rounds),
			ray_init_args=ray_init_args,
			strategy=strategy,
		)
	except Exception as _:
		wandb.finish(exit_code=1)

	wandb.finish()


if __name__ == '__main__':
	main()
