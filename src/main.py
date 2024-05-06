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

parser.add_argument(
	"--capturing",
	action=argparse.BooleanOptionalAction,
	help="Whether to save the messages from client to server"
)


def main():
	args = parser.parse_args()

	client_resources = {
		"num_cpus": args.num_cpus,
		"num_gpus": args.num_gpus
	}

	configs = batch_config.generate_configs_from_yaml_file(str(Path(args.yaml_file).resolve()))

	run_name = args.run_name
	is_online = args.logging
	is_capturing = args.capturing

	log(INFO, f"Loaded {len(configs)} configs with name {run_name}, running...")
	for config in configs:
		config.run_simulation(client_resources, is_online, is_capturing, run_name)

if __name__ == '__main__':
	main()
