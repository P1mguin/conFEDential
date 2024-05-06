import os
import pickle
from typing import Callable, List, Tuple

import numpy as np
import torch
from fedartml import SplitAsFederatedData
from torch.utils.data import DataLoader, random_split, TensorDataset

from src.datasets.Dataset import Dataset


class Texas(Dataset):
	@staticmethod
	def load_data(
			client_count: int,
			batch_size: int,
			preprocess_fn: Callable[[dict], dict],
			alpha: float | None = None,
			percent_non_iid: float | None = None,
			seed: int = 78,
			function_hash: str = ""
	) -> Tuple[List[DataLoader], DataLoader]:
		cache_file = f".cache/preprocessed/texas/{seed}{alpha}{percent_non_iid}{client_count}{batch_size}/{function_hash}"
		if os.path.exists(cache_file):
			with open(cache_file, "rb") as f:
				return pickle.load(f)

		Dataset.is_data_downloaded("texas")

		texas = np.load(".cache/texas/texas100.npz")
		features = torch.from_numpy(texas["features"])
		labels = torch.from_numpy(np.argmax(texas["labels"], axis=1))

		# Split the features and labels into a train and test set in the same way
		dataset = TensorDataset(features, labels)
		train_size = int(0.85 * len(dataset))
		test_size = len(dataset) - train_size

		train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

		# If filled, split the data non-iid
		if alpha is not None and percent_non_iid is not None:
			federated_train_data, _, _, _ = SplitAsFederatedData(random_state=seed).create_clients(
				image_list=features[train_dataset.indices],
				label_list=labels[train_dataset.indices],
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
			clients = ([(value[0], value[1]) for value in subset] for subset in subsets)

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
