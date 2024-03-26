import argparse
import math
import multiprocessing as mp
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Type

import flwr as fl
import torch
import torch.nn as nn
from flwr.common import Scalar
from torch.utils.data import DataLoader

import helper
import src.utils as utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description="Running conFEDential simulation")
seed = 78

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
		trainloaders: List[DataLoader],
		epochs: int,
		model_class: Type[nn.Module],
		criterion_class: Type[nn.Module],
		optimizer_class: Type[torch.optim.Optimizer]
) -> Callable[[str], fl.client.Client]:
	class FlowerClient(fl.client.NumPyClient):
		# Client does not get evaluation method, that is done at server level over all data at once
		def __init__(self, cid: int):
			self.parameters = None
			self.trainloader = trainloaders[cid]

		def get_parameters(self, config):
			return self.parameters

		def set_parameters(self, parameters):
			self.parameters = parameters

		def fit(self, parameters, config):
			self.set_parameters(parameters)

			new_parameters, data_size = helper.train(epochs, parameters, model_class, self.trainloader, criterion_class,
													 optimizer_class)

			return new_parameters, data_size, {}

	def client_fn(cid: str) -> fl.client.Client:
		cid = int(cid)
		return FlowerClient(cid).to_client()

	return client_fn


def get_evaluate_fn(
		testloader: DataLoader,
		model_class: Type[nn.Module],
		criterion_class: Type[nn.Module]
) -> Callable[[int, fl.common.NDArrays, Dict[str, Scalar]], Tuple[float, Dict[str, Scalar]]]:
	def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]):
		loss, accuracy, data_size = helper.test(parameters, model_class, testloader, criterion_class)

		return loss, {"accuracy": accuracy, "data_size": data_size, "server_round": server_round}

	return evaluate


def main() -> None:
	args = parser.parse_args()

	mp.set_start_method("spawn")

	client_resources = {
		"num_cpus": args.num_cpus,
		"num_gpus": args.num_gpus
	}

	yaml_file = str(Path(args.yaml_file).resolve())
	config = utils.load_yaml_file(yaml_file)

	model_class = utils.load_model(yaml_file)
	criterion_class = getattr(torch.nn, config["model"]["criterion"]["type"])
	optimizer_class = getattr(torch.optim, config["simulation"]["learning_method"]["optimizer"])
	trainloaders, testloader = utils.load_dataloaders(config)
	epochs = config["simulation"]["local_rounds"]
	global_rounds = config["simulation"]["global_rounds"]

	client_fn = get_client_fn(trainloaders, epochs, model_class, criterion_class, optimizer_class)

	client_count = config["simulation"]["client_count"]

	fraction_fit = config["simulation"]["fraction_fit"]
	fraction_evaluate = 0.0
	min_fit_clients = max(math.floor(fraction_fit * client_count), 1)
	min_evaluate_clients = 0
	evaluate = get_evaluate_fn(testloader, model_class, criterion_class)
	initial_parameters = fl.common.ndarrays_to_parameters(helper.get_weights(model_class()))

	strategy = fl.server.strategy.FedAvg(
		fraction_fit=fraction_fit,
		fraction_evaluate=fraction_evaluate,
		min_fit_clients=min_fit_clients,
		min_evaluate_clients=min_evaluate_clients,
		min_available_clients=min_fit_clients,
		evaluate_fn=evaluate,
		initial_parameters=initial_parameters
	)

	fl.simulation.start_simulation(
		client_fn=client_fn,
		num_clients=client_count,
		client_resources=client_resources,
		config=fl.server.ServerConfig(num_rounds=global_rounds),
		strategy=strategy
	)


if __name__ == '__main__':
	main()
