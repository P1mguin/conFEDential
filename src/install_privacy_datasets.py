import argparse
import asyncio
import os
import tarfile

import httpx
import pandas as pd
from tqdm import tqdm

datasets = {
	"purchase": {
		"link": "https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz",
		"cache_path": "purchase/purchase/",
		"target_path": "purchase/purchase/purchase.parquet"
	},
	"texas": {
		"link": "https://www.comp.nus.edu.sg/~reza/files/dataset_texas.tgz",
		"cache_path": "texas/texas/",
		"target_path": "texas/texas/texas.parquet"
	}
}


def uncompress_tgz(name: str, cache_root: str):
	"""
	Uncompresses a tgz file into a directory
	:param name: the name of the dataset in the tgz file
	:param cache_root: the root directory of the cache
	"""
	print(f"Uncompressing {name} dataset")

	# Get the paths in which the tgz file is stored and the directory to extract to
	dataset = datasets[name]
	cache_dir = f"{cache_root}data/{dataset['cache_path']}"
	file_path = f"{cache_root}data/{dataset['cache_path']}{name}.tgz"

	# If there is more than one file in the cache dir continue
	if len(os.listdir(cache_dir)) > 1:
		print(f"Files already extracted: {cache_dir}")
		return

	# Extract the tgz file
	with tarfile.open(file_path, "r:gz") as tar:
		tar.extractall(cache_dir)


async def download_files(cache_root: str):
	"""
	Download all the files in the datasets dictionary
	:param cache_root: the root directory of the cache
	"""
	# Create an event loop and download all the files
	loop = asyncio.get_running_loop()
	urls = [dataset["link"] for name, dataset in datasets.items()]
	file_paths = [f"{cache_root}data/{dataset['cache_path']}{name}.tgz" for name, dataset in datasets.items()]
	args = zip(urls, file_paths)
	tasks = [loop.create_task(download_file(*arg)) for arg in args]
	await asyncio.gather(*tasks, return_exceptions=True)


async def download_file(url: str, file_path: str):
	"""
	Download a file from a URL to a file path
	:param url: the url to download the file from
	:param file_path: the path to save the file to
	"""
	# If the file already exists continue
	if os.path.exists(file_path):
		print(f"File already exists: {file_path}")
		return

	# Download the file
	os.makedirs(os.path.dirname(file_path), exist_ok=True)
	with open(file_path, "wb") as f:
		async with httpx.AsyncClient() as client:
			async with client.stream("GET", url) as r:
				r.raise_for_status()
				total = int(r.headers.get("content-length", 0))
				tqdm_params = {
					"desc": url,
					"total": total,
					"unit": "B",
					"unit_scale": True,
					"unit_divisor": 1024,
				}
				with tqdm(**tqdm_params) as pbar:
					downloaded = r.num_bytes_downloaded
					async for chunk in r.aiter_bytes():
						pbar.update(r.num_bytes_downloaded - downloaded)
						f.write(chunk)
						downloaded = r.num_bytes_downloaded


def process_purchase(cache_root: str):
	"""
	Process the purchase dataset into a parquet file
	:param cache_root: the root directory of the cache
	"""
	# Set the configuration of where to load and store the data
	data_config = datasets["purchase"]
	cache_path = f"{cache_root}data/{data_config['cache_path']}dataset_purchase"
	parquet_path = f"{cache_root}data/{data_config['target_path']}"

	# If arrow file exists continue
	if os.path.exists(parquet_path):
		print(f"Arrow file already exists: {parquet_path}")
		return

	# Load the file into a pandas dataframe
	print(f"Loading purchase CSV to dataframe")
	column_names = ["C_" + str(i) for i in range(601)]
	df = pd.read_csv(cache_path, names=column_names, header=None)

	# Convert the first column to an int64 and decrement by 1, and convert other values to float32
	print(f"Converting to int and floats")
	df.iloc[:, 0] = df.iloc[:, 0].astype("int64") - 1
	df.iloc[:, 1:] = df.iloc[:, 1:].astype("float32")

	# Store the dataframe as a parquet file
	print("Storing dataframe as parquet file")
	df.to_parquet(parquet_path, compression="snappy")


def process_texas(cache_root: str):
	"""
	Process the texas dataset into a parquet file
	:param cache_root: the root directory of the cache
	"""
	# Set the configuration of where to load and store the data
	data_config = datasets["texas"]
	texas_100_path = f"{cache_root}data/{data_config['cache_path']}texas/100/"
	parquet_path = f"{cache_root}data/{data_config['target_path']}"
	features_path = f"{texas_100_path}feats"
	labels_path = f"{texas_100_path}labels"

	# Load the features and labels into pandas dataframes
	print("Loading texas features and labels CSV to dataframe")
	column_names = ["C_" + str(i) for i in range(6170)]
	df_features = pd.read_csv(features_path, names=column_names[1:], header=None, dtype="float32")
	df_labels = pd.read_csv(labels_path, names=[column_names[0]], header=None, dtype="int64") - 1

	# Create one dataframe with labels as first column and features as the rest
	print("Creating single dataframe")
	df = pd.concat([df_labels, df_features], axis=1)

	# Store the dataframe as a parquet file
	print("Storing dataframe as parquet file")
	df.to_parquet(parquet_path, compression="snappy")


def clean_up_directories(cache_root: str):
	"""
	Clean up the directories by removing all files except the arrow files
	:param cache_root: the root directory of the cache
	"""
	# Walk the cache directory and remove everything except the arrow file
	for name in datasets.keys():
		print(f"Cleaning up {name} directory")
		data_config = datasets[name]
		cache_dir = f"{cache_root}data/{data_config['cache_path']}"

		# Walk the cache directory leaf to root and remove everything except the arrow file, including directories
		for root, dirs, files in os.walk(cache_dir, topdown=False):
			for file in files:
				if file.endswith(".parquet"):
					continue
				os.remove(os.path.join(root, file))
			for dir in dirs:
				os.rmdir(os.path.join(root, dir))


# Arguments for the attack simulation
parser = argparse.ArgumentParser(description="Download PURCHASE100 and TEXAS100 from Shokris repositories")
parser.add_argument(
	"--cache-root",
	type=str,
	default="./.cache/",
	help="Absolute path to root of the directory in which the model architecture will be saved"
)


def main():
	# Get the run arguments
	args = parser.parse_args()
	cache_root = f"{os.path.abspath(args.cache_root)}/"

	# Download the datasets
	asyncio.run(download_files(cache_root))

	# Uncompress the datasets
	uncompress_tgz("purchase", cache_root)
	uncompress_tgz("texas", cache_root)

	# Preprocess the datasets
	process_purchase(cache_root)
	process_texas(cache_root)

	# Clean up the directories
	clean_up_directories(cache_root)


if __name__ == '__main__':
	main()
