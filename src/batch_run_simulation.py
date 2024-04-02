import os
import sys

PROJECT_DIRECTORY = os.path.abspath(os.path.join(os.getcwd(), "./"))
sys.path.append(PROJECT_DIRECTORY)

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from src import utils
from src.run_simulation import run_simulation

torch.manual_seed(78)
random.seed(78)
np.random.seed(78)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def main() -> None:
	args = parser.parse_args()

	client_resources = {
		"num_cpus": args.num_cpus,
		"num_gpus": args.num_gpus
	}

	yaml_file = str(Path(args.yaml_file).resolve())
	batch_config = utils.load_yaml_file(yaml_file)
	configs = utils.load_configs_from_batch_config_path(yaml_file)
	print(f"Loaded {len(configs)} configs, running...")
	batch_run_name = f"{batch_config['dataset']['name']}-{batch_config['model']['name']}-{batch_config['simulation']['learning_method']['optimizer']}"
	for config in configs:
		print(config)
		run_simulation(config, client_resources, batch_run_name)


if __name__ == '__main__':
	main()
