import argparse
from pathlib import Path

import torch

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


def main() -> None:
	args = parser.parse_args()

	client_resources = {
		"num_cpus": args.num_cpus,
		"num_gpus": args.num_gpus
	}

	yaml_file = str(Path(args.yaml_file).resolve())
	config = utils.load_yaml_file(yaml_file)

	model_class = utils.load_model(yaml_file)
	criterion = getattr(torch.nn, config["model"]["criterion"]["type"])
	optimizer = getattr(torch.optim, config["simulation"]["learning_method"]["optimizer"])
	dataloaders, testloader = utils.load_dataloaders(config)


if __name__ == '__main__':
	main()
