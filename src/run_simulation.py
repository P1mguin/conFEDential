import argparse
from pathlib import Path

import torch

import federated_datasets
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


def main():
	args = parser.parse_args()

	client_resources = {
		"num_cpus": args.num_cpus,
		"num_gpus": args.num_gpus
	}

	yaml_file = Path(args.yaml_file).resolve()
	config = utils.load_yaml_file(yaml_file)

	client_count = config["simulation"]["client_count"]
	batch_size = config["simulation"]["batch_size"]
	preprocess_fn = utils.load_func(config["dataset"]["preprocess_fn"])
	alpha = config["dataset"]["splitter"]["alpha"]
	percent_noniid = config["dataset"]["splitter"]["percent_noniid"]
	dataclass = getattr(federated_datasets, config["dataset"]["name"])
	dataloaders, testloader = dataclass.load_data(client_count, batch_size, preprocess_fn, alpha, percent_noniid)


if __name__ == '__main__':
	main()
