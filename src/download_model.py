import argparse
import os

import torch

parser = argparse.ArgumentParser(description="Download module from torch hub in correct cache directory")

parser.add_argument(
	"--repo",
	type=str,
	help="The torch hub repository name"
)

parser.add_argument(
	"--model",
	type=str,
	help="The torch hub model name"
)

parser.add_argument(
	"--cache-root",
	type=str,
	default="./.cache/",
	help="Absolute path to root of the directory in which the model architecture will be saved"
)


def main() -> None:
	args = parser.parse_args()
	repo = args.repo
	model = args.model
	net = torch.hub.load(repo, model, force_reload=True)
	cache_root = f"{os.path.abspath(args.cache_root)}/"
	os.makedirs(f"{cache_root}model_architectures/", exist_ok=True)
	torch.save(net, f"{cache_root}model_architectures/{repo.replace('/', '')}_{model}.pth")


if __name__ == '__main__':
	main()
