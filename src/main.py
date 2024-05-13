import sys
import os

# Keep at top, so cluster knows which directory to work in
PROJECT_DIRECTORY = os.path.abspath(os.path.join(os.getcwd(), "./"))
sys.path.append(PROJECT_DIRECTORY)

import argparse
import random
from logging import INFO
from pathlib import Path

import numpy as np
import torch
from flwr.common import log

from src.utils import batch_config

torch.manual_seed(78)
random.seed(78)
np.random.seed(78)

parser = argparse.ArgumentParser(description="Running conFEDential simulation for a variety of configurations")

parser.add_argument(
	"--yaml-file",
	type=str,
	help="Path to the yaml file that contains the general configuration of the simulation"
)

parser.add_argument(
	"--clients",
	type=int,
	help="The amount of concurrent clients to run simultaneously"
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

parser.add_argument(
	"--capturing",
	action=argparse.BooleanOptionalAction,
	help="Whether to save the messages from client to server"
)


def main():
	args = parser.parse_args()

	configs = batch_config.generate_configs_from_yaml_file(str(Path(args.yaml_file).resolve()))

	concurrent_clients = args.clients
	run_name = args.run_name
	is_online = args.logging
	is_capturing = args.capturing

	log(INFO, f"Loaded {len(configs)} configs with name {run_name}, running...")
	for config in configs:
		config.run_simulation(concurrent_clients, is_online, is_capturing, run_name)

if __name__ == '__main__':
	main()
