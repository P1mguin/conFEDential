import os
import pickle
from typing import Callable, List, Tuple

from datasets import load_dataset
from fedartml import SplitAsFederatedData
from torch.utils.data import DataLoader, random_split

from src.federated_datasets.Dataset import Dataset


class ImageNet(Dataset):
	@staticmethod
	def load_data(
			client_count: int,
			batch_size: int,
			preprocess_fn: Callable[[dict], dict],
			alpha: float | None = None,
			percent_non_iid: float | None = None,
			seed: int = 78,
			function_hash: str = "",
	) -> Tuple[List[DataLoader], DataLoader]:
		"""
		Load data is called for all datasets such that the train data loaders and test data loader is returned.
		The list of train data loaders is of size client_count and represents the data of the clients in the federation,
		the data is drawn non-iid via FedArtML. Evaluation is done centrally and so this function only returns one test
		data loader with all test data.
		:param client_count: The amount of clients in the federation
		:param batch_size: The batch size, -1 means all data at once
		:param preprocess_fn: The function that is applied to each element to preprocess
		:param alpha: Concentration parameter of the Dirichlet distribution defining the desired degree of non-IID-ness
		for the labels of the federated data
		:param percent_non_iid: Percentage (between 0 and 100) desired of non-IID-ness for the labels of the federated data
		:param seed: Controls the starting point of the random number generator. Value is passed for reproducibility,
		use your favourite number.
		:param function_hash: The hash of the function that is used to cache the preprocessed data for this function
		"""
		# See if the information has been cached before and if so load from cache
		cache_file = f".cache/preprocessed/imagenet/{seed}{alpha}{percent_non_iid}{client_count}{batch_size}/{function_hash}"
		if os.path.exists(cache_file):
			# Load the data from the cache
			with open(cache_file, "rb") as f:
				return pickle.load(f)

		# Confirm the dataset is downloaded locally and load in the dataset
		Dataset.is_data_downloaded("zh-plus___tiny-imagenet")
		train_dataset, test_dataset = load_dataset(
			"zh-plus/tiny-imagenet",
			name="default",
			cache_dir=".cache",
			split=["train", "valid"],
			download_mode="reuse_dataset_if_exists"
		)

		# Convert the pytorch tensor to NumPy such FedArtML can convert the data to non-iid
		train_dataset.set_format(type="np")
		test_dataset.set_format(type="np")

		# Apply the received preprocess function
		train_dataset = train_dataset.map(preprocess_fn)
		test_dataset = test_dataset.map(preprocess_fn)

		# If filled, split the data non-iid
		if alpha is not None and percent_non_iid is not None:
			federated_train_data, _, _, _ = SplitAsFederatedData(random_state=seed).create_clients(
				image_list=train_dataset["x"],
				label_list=train_dataset["y"],
				num_clients=client_count,
				method="dirichlet",
				alpha=alpha,
				percent_noniid=percent_non_iid
			)
			clients = federated_train_data["with_class_completion"].values()
		else:
			lengths = [len(train_dataset) // client_count] * client_count
			lengths[0] += len(train_dataset) % client_count
			subsets = random_split(train_dataset, lengths)
			clients = ([(value["x"], value["y"]) for value in subset] for subset in subsets)

		# Use with class completion so every client has at least one label of each class
		# Create the train and test loaders and return
		train_loaders = []
		for client_data in clients:
			# Batch_size -1 corresponds to infinity
			if batch_size == -1:
				batch_size = len(client_data)
			data_loader = DataLoader(client_data, batch_size=batch_size)
			train_loaders.append(data_loader)
		test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

		# Cache (train_loaders, test_loader) to the cache file
		os.makedirs(os.path.dirname(cache_file), exist_ok=True)
		with open(cache_file, "wb") as f:
			pickle.dump((train_loaders, test_loader), f)

		return train_loaders, test_loader
