import argparse
import os
import random
from logging import INFO
from pathlib import Path

import numpy as np
import torch.autograd
from flwr.common.logger import log

from src.utils.configs import AttackConfig

os.environ["HF_DATASETS_OFFLINE"] = "1"
torch.manual_seed(78)
random.seed(78)
np.random.seed(78)
torch.autograd.set_detect_anomaly(True)

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

	# Train the attacker model with the known information

	# Test the model
	return None

def main():
	args = parser.parse_args()

	client_resources = {
		"num_cpus": args.num_cpus,
		"num_gpus": args.num_gpus
	}

	config = AttackConfig.from_yaml_file(str(Path(args.yaml_file).resolve()))
	run_name = args.run_name
	is_online = args.logging

	log(INFO, "Running attack simulation")
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
