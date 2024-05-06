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


def main() -> None:
	args = parser.parse_args()
	repo = args.repo
	model = args.model
	net = torch.hub.load(repo, model, force_reload=True)
	os.makedirs(".cache/models/", exist_ok=True)
	torch.save(net, f".cache/models/{repo.replace('/', '')}_{model}.pth")


if __name__ == '__main__':
	main()
