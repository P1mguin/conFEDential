import os
import sys

# Keep at top, so cluster knows which directory to work in
PROJECT_DIRECTORY = os.path.abspath(os.path.join(os.getcwd(), "./"))
sys.path.append(PROJECT_DIRECTORY)

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
import wandb
from flwr.common.logger import log
from torch.utils.data import DataLoader

from src import training
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


def attack_simulation(config: AttackConfig, args: argparse.Namespace) -> None:
	# Get the information that we can use in the attack
	# TODO: Use in attack
	simulation_aggregates = config.get_model_aggregates()
	client_updates = config.get_client_updates()

	# Construct the attacker model based on the config
	log(INFO, "Constructing attacker model dataset")
	attack_loader = get_attack_dataset(config)

	# 80 percent train, 10 percent test, 10 percent validation
	train_loader, test_loader = split_dataloader(attack_loader, 0.9)
	train_loader, validation_loader = split_dataloader(train_loader, 0.9)

	# Take the model, BCELoss since we work with one probability, and the optimizer
	attack_model = config.get_attack_model().to(training.DEVICE)
	criterion = torch.nn.BCELoss()
	optimizer = config.get_attack_optimizer(attack_model.parameters())

	wandb_kwargs = config.get_wandb_kwargs(f"attack_{args.run_name}")
	mode = "online" if args.logging else "offline"
	wandb.init(mode=mode, **wandb_kwargs)

	# Train the attack model until convergence
	log(INFO, "Constructed dataset, starting training")
	previous_loss, previous_accuracy = test_attack_model(criterion, attack_model, validation_loader)

	i = -1
	try:
		while True:
			i += 1
			correct, total, train_loss = 0, 0, 0.
			for parameters, data, target, is_member in train_loader:
				data, target, is_member = data.to(device), target.to(device), is_member.to(device)
				optimizer.zero_grad()
				output = attack_model(parameters, data, target)
				loss = criterion(output, is_member.float())
				loss.backward()
				optimizer.step()

				train_loss += loss.item()
				total += is_member.size()[0]
				correct += (output.round() == is_member).sum().item()

			train_loss /= total
			train_accuracy = correct / total
			validation_loss, validation_accuracy = test_attack_model(criterion, attack_model, validation_loader)
			test_loss, test_accuracy = test_attack_model(criterion, attack_model, test_loader)

			log_string = f"Finished epoch {i}:\n"
			log_string += f"Train loss: {train_loss}, Train accuracy: {train_accuracy}\n"
			log_string += f"Validation loss: {validation_loss}, Validation accuracy: {validation_accuracy}\n"
			log_string += f"Test loss: {test_loss}, Test accuracy: {test_accuracy}"
			log(INFO, log_string)

			wandb.log({
				"train_loss": train_loss,
				"train_accuracy": train_accuracy,
				"validation_loss": validation_loss,
				"validation_accuracy": validation_accuracy,
				"test_loss": test_loss,
				"test_accuracy": test_accuracy
			})

			if validation_loss > previous_loss and validation_accuracy < previous_accuracy:
				break

			previous_loss, previous_accuracy = validation_loss, validation_accuracy
	except Exception as _:
		wandb.finish(exit_code=1)

	wandb.finish()

	# Test the model
	return None


def test_attack_model(criterion: nn.Module, attack_model: nn.Module, data_loader: DataLoader) -> Tuple[float, float]:
	attack_model.eval()
	correct, total, loss = 0, 0, 0.
	for parameters, data, target, is_member in data_loader:
		data, target, is_member = data.to(device), target.to(device), is_member.to(device)
		output = attack_model(parameters, data, target).round()

		with torch.no_grad():
			loss += criterion(output, is_member.float()).item()

		total += is_member.size()[0]
		correct += (output == is_member).sum().item()

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
		for i, (train_loader, test_loader) in enumerate(data_loaders):
			log(INFO, f"Training shadow model {i}")
			parameters = train_shadow_model(config, train_loader)
			shadow_models.append((parameters, train_loader, test_loader))

		# Cache the results for the shadow models so that we can more easily tune the attack model
		os.makedirs("/".join(cache_file.split("/")[:-1]), exist_ok=True)
		with open(cache_file, "wb") as file:
			pickle.dump(shadow_models, file)

	# Generate the dataset on which to train the attack model
	dataset = []
	for parameters, train_loader, test_loader in shadow_models:
		for data, target in train_loader.dataset:
			dataset.append((parameters, data, target, 1))

		for data, target in test_loader.dataset:
			dataset.append((parameters, data, target, 0))

	batch_size = config.get_attack_batch_size()
	return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_shadow_model(run_config: AttackConfig, train_loader: DataLoader):
	strategy = run_config.get_strategy()
	global_rounds = run_config.get_global_rounds()
	parameters = None
	config = {}
	for i in range(global_rounds):
		# TODO: Maintain intermediate state
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
	attack_simulation(config, args)


if __name__ == '__main__':
	main()
