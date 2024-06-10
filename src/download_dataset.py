import argparse
import os

from datasets import load_dataset

parser = argparse.ArgumentParser(description="Download dataset from hugging face in correct cache directory")

parser.add_argument(
	"--dataset",
	type=str,
	help="The Hugging Face dataset name"
)

parser.add_argument(
	"--cache-root",
	type=str,
	default="./.cache/",
	help="Absolute path to root of the directory in which the model architecture will be saved"
)


def main() -> None:
	args = parser.parse_args()
	cache_root = f"{os.path.abspath(args.cache_root)}/"
	dataset = args.dataset
	load_dataset(dataset, cache_dir=f"{cache_root}data", download_mode="force_redownload", split=["train", "test"])


if __name__ == '__main__':
	main()
