import os
import sys
from datetime import datetime

PROJECT_DIRECTORY = os.path.abspath(os.path.join(os.getcwd(), "./"))
sys.path.append(PROJECT_DIRECTORY)

import argparse
import random
from pathlib import Path
from logging import INFO

from flwr.common.logger import log
import numpy as np
import torch
import src.utils.batch_config as batch_config

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

	configs = batch_config.generate_configs_from_yaml_file(str(Path(args.yaml_file).resolve()))
	dataset_name = configs[0].get_dataset_name()
	model_name = configs[0].get_model_name()
	optimizer = configs[0].get_optimizer_name()
	start_day = datetime.now().strftime("%Y-%m-%d")
	batch_run_name = f"{dataset_name}-{model_name}-{optimizer}-{start_day}"

	log(INFO, f"Loaded {len(configs)} configs, running...")
	for config in configs:
		run_simulation(config, client_resources, batch_run_name)


if __name__ == '__main__':
	main()
