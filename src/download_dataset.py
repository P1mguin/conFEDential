import argparse

from datasets import load_dataset

parser = argparse.ArgumentParser(description="Download dataset from hugging face in correct cache directory")

parser.add_argument(
	"--dataset",
	type=str,
	help="The Hugging Face dataset name"
)


def main() -> None:
	args = parser.parse_args()
	dataset = args.dataset
	load_dataset(dataset, cache_dir=".cache/data", download_mode="force_redownload", split=["train", "test"])


if __name__ == '__main__':
	main()
