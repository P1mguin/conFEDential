import argparse
import os
import pickle
import random
from logging import INFO
from pathlib import Path
from typing import Tuple

import numpy as np
import torch.autograd
import torch.nn as nn
from flwr.common.logger import log
from torch.utils.data import DataLoader

from src.simulations.performance.run_simulation import run_simulation
from src.utils import split_dataloader
from src.utils.configs import AttackConfig

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

parser.add_argument(
	"--run-name",
	type=str,
	default=None,
	help="Name of the run that will be added as tag to the Weights and Biases dashboard"
)

parser.add_argument(
	"--logging",
	action=argparse.BooleanOptionalAction,
	help="Whether to log to Weights and Biases dashboard during simulation"
)


def attack_simulation(
		config: AttackConfig,
) -> None:
	# Get the information that we can use in the attack
	simulation_aggregates = config.get_model_aggregates()
	client_updates = config.get_client_updates()

	# Construct the attacker model based on the config
	attack_loader = get_attack_dataset(config)

	# 80 percent train, 10 percent test, 10 percent validation
	train_loader, test_loader = split_dataloader(attack_loader, 0.9)
	train_loader, validation_loader = split_dataloader(train_loader, 0.9)

	# Take the model, BCELoss since we work with one probability, and the optimizer
	attack_model = config.get_attack_model()
	criterion = torch.nn.BCELoss()
	optimizer = config.get_attack_optimizer(attack_model.parameters())

	# Train the attack model until convergence
	parameters = None
	loss, accuracy = test_attack_model(criterion, attack_model, validation_loader)

	# Test the model
	return None


def test_attack_model(criterion: nn.Module, attack_model: nn.Module, data_loader: DataLoader) -> Tuple[float, float]:
	attack_model.eval()
	correct, total, loss = 0, 0, 0.

	for model, data, target, is_train in data_loader:
		model = [layer[0] for layer in model]
		data, target, is_train = data[0], target[0], is_train[0]
		output = attack_model(model, data, target)[0]

		with torch.no_grad():
			loss += criterion(output, is_train.float()).item()

		total += 1
		correct += (output == target).sum().item()

	accuracy = correct / total
	loss /= total
	return loss, accuracy


def get_attack_dataset(config: AttackConfig) -> DataLoader:
	# Get the data split to train the several models
	data_loaders = config.get_attack_data_loaders()

	cache_file = f".shadow_models/{'/'.join(config.get_output_capture_directory_path().split('/')[1:])[:-1]}.pkl"
	if os.path.exists(cache_file):
		# Load the data from the cache
		with open(cache_file, "rb") as f:
			shadow_models = pickle.load(f)
	else:
		# Train all the shadow models for the dataloaders
		shadow_models = []
		for train_loader, test_loader in data_loaders:
			model = train_shadow_model(config, train_loader)
			shadow_models.append((model, train_loader, test_loader))

		# Cache the results for the shadow models so that we can more easily tune the attack model
		os.makedirs("/".join(cache_file.split("/")[:-1]), exist_ok=True)
		with open(cache_file, "wb") as file:
			pickle.dump(shadow_models, file)

	# Generate the dataset on which to train the attack model
	dataset = []
	for model, train_loader, test_loader in shadow_models:
		for data, target in train_loader.dataset:
			dataset.append((model, data, target, 1))

		for data, target in test_loader.dataset:
			dataset.append((model, data, target, 0))

	batch_size = config.get_attack_batch_size()
	return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_shadow_model(run_config: AttackConfig, train_loader: DataLoader):
	strategy = run_config.get_strategy()
	global_rounds = run_config.get_global_rounds()
	parameters = None
	config = {}
	for i in range(global_rounds):
		# TODO: Maintain intermediate state
		print("Starting round", i)
		parameters, _, config = strategy.train(parameters, train_loader, run_config, config)

	# TODO: Also attack using config
	return parameters

def main():
	args = parser.parse_args()

	client_resources = {
		"num_cpus": args.num_cpus,
		"num_gpus": args.num_gpus
	}

	config = AttackConfig.from_yaml_file(str(Path(args.yaml_file).resolve()))
	run_name = args.run_name
	is_online = args.logging

	capture_output_path = config.get_output_capture_directory_path()
	if not os.path.exists(capture_output_path):
		log(INFO, f"Captured parameters not found, running training simulation")
		run_simulation(
			config,
			client_resources,
			run_name,
			is_online,
			is_capturing=True
		)

	log(INFO, "Finished training, starting attack simulation")
	attack_simulation(config)


if __name__ == '__main__':
	main()
