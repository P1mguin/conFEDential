import asyncio
import os
import tarfile

import httpx
import pandas as pd
import pyarrow as pa
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


def uncompress_tgz(name: str):
	print(f"Uncompressing {name} dataset")
	dataset = datasets[name]
	cache_dir = f".cache/data/{dataset['cache_path']}"
	file_path = f".cache/data/{dataset['cache_path']}{name}.tgz"

	# If there is more than one file in the cache dir continue
	if len(os.listdir(cache_dir)) > 1:
		print(f"Files already extracted: {cache_dir}")
		return

	with tarfile.open(file_path, "r:gz") as tar:
		tar.extractall(cache_dir)


async def download_files():
	loop = asyncio.get_running_loop()

	urls = [dataset["link"] for name, dataset in datasets.items()]
	file_paths = [f".cache/data/{dataset['cache_path']}{name}.tgz" for name, dataset in datasets.items()]
	args = zip(urls, file_paths)
	tasks = [loop.create_task(download_file(*arg)) for arg in args]
	await asyncio.gather(*tasks, return_exceptions=True)


async def download_file(url: str, file_path: str):
	if os.path.exists(file_path):
		print(f"File already exists: {file_path}")
		return

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


def process_purchase():
	data_config = datasets["purchase"]
	cache_path = f".cache/data/{data_config['cache_path']}dataset_purchase"
	parquet_path = f".cache/data/{data_config['target_path']}"

	# If arrow file exists continue
	if os.path.exists(parquet_path):
		print(f"Arrow file already exists: {parquet_path}")
		return

	# Load the file into a pandas dataframe
	print(f"Loading purchase CSV to dataframe")
	column_names = ["C_" + str(i) for i in range(601)]
	df = pd.read_csv(cache_path, names=column_names, header=None)

	# Convert the first column to an int32 and decrement by 1, and convert other values to float64
	print(f"Converting to int and floats")
	df.iloc[:, 0] = df.iloc[:, 0].astype("int32") - 1
	df.iloc[:, 1:] = df.iloc[:, 1:].astype("float64")

	# Store the dataframe as a parquet file
	print("Storing dataframe as parquet file")
	df.to_parquet(parquet_path, compression="snappy")

def process_texas():
	data_config = datasets["texas"]
	texas_100_path = f".cache/data/{data_config['cache_path']}texas/100/"
	parquet_path = f".cache/data/{data_config['target_path']}"
	features_path = f"{texas_100_path}feats"
	labels_path = f"{texas_100_path}labels"

	# Load the features and labels into pandas dataframes
	print("Loading texas features and labels CSV to dataframe")
	column_names = ["C_" + str(i) for i in range(6170)]
	df_features = pd.read_csv(features_path, names=column_names[1:], header=None, dtype="int32") - 1
	df_labels = pd.read_csv(labels_path, names=[column_names[0]], header=None, dtype="float64")

	# Create one dataframe with labels as first column and features as the rest
	print("Creating single dataframe")
	df = pd.concat([df_labels, df_features], axis=1)

	# Store the dataframe as a parquet file
	print("Storing dataframe as parquet file")
	df.to_parquet(parquet_path, compression="snappy")

def clean_up_directories():
	for name in datasets.keys():
		print(f"Cleaning up {name} directory")
		data_config = datasets[name]
		cache_dir = f".cache/data/{data_config['cache_path']}"

		# Walk the cache directory leaf to root and remove everything except the arrow file, including directories
		for root, dirs, files in os.walk(cache_dir, topdown=False):
			for file in files:
				if file.endswith(".parquet"):
					continue
				os.remove(os.path.join(root, file))
			for dir in dirs:
				os.rmdir(os.path.join(root, dir))


def main():
	asyncio.run(download_files())

	uncompress_tgz("purchase")
	uncompress_tgz("texas")

	process_purchase()
	process_texas()

	clean_up_directories()


if __name__ == '__main__':
	main()
