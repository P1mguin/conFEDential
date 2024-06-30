import os
import sys

# Keep at top, so cluster knows which directory to work in
PROJECT_DIRECTORY = os.path.abspath(os.path.join(os.getcwd(), "./"))
sys.path.append(PROJECT_DIRECTORY)

import argparse
import random
from logging import INFO, ERROR
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
	default=4,
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
	"--ray",
	default=False,
	action=argparse.BooleanOptionalAction,
	help="Whether ray has been initialised already"
)

parser.add_argument(
	"--logging",
	default=False,
	action=argparse.BooleanOptionalAction,
	help="Whether to log to Weights and Biases dashboard during simulation"
)

parser.add_argument(
	"--capturing",
	default=True,
	action=argparse.BooleanOptionalAction,
	help="Whether to save the messages from client to server"
)

parser.add_argument(
	"--memory",
	type=float,
	default=None,
	help="Amount of memory to allocate to the simulation in GB, by default takes all memory available"
)

parser.add_argument(
	"--num-cpus",
	type=int,
	default=None,
	help="Number of CPUs to allocate to the simulation, by default takes all CPUs available"
)

parser.add_argument(
	"--num-gpus",
	type=int,
	default=None,
	help="Number of GPUs to allocate to the simulation, by default takes all GPUs available"
)

parser.add_argument(
	"--cache-root",
	type=str,
	default="./.cache/",
	help="Absolute path to root of the directory in which the datasets, model architectures, experiment results as well as partial computations will be stored to prevent recomputing heavy experiments"
)


def main():
	args = parser.parse_args()

	concurrent_clients = args.clients
	memory = args.memory
	num_cpus = args.num_cpus
	num_gpus = args.num_gpus
	is_ray_initialised = args.ray
	is_online = args.logging
	is_capturing = args.capturing
	run_name = args.run_name
	cache_root = f"{os.path.abspath(args.cache_root)}/"

	configs = batch_config.generate_configs_from_yaml_file(str(Path(args.yaml_file).resolve()), cache_root)

	log(INFO, f"Loaded {len(configs)} configs with name {run_name}, running...")
	for config in configs:
		log(INFO, config)
		try:
			config.run_simulation(
				concurrent_clients,
				memory,
				num_cpus,
				num_gpus,
				is_ray_initialised,
				is_online,
				is_capturing,
				run_name,
			)
		except Exception as e:
			log(ERROR, f"Failed to run simulation {e}")


if __name__ == '__main__':
	main()
