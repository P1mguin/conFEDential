from typing import Callable, List, Tuple

from datasets import load_dataset
from fedartml import SplitAsFederatedData
from torch.utils.data import DataLoader

from src.federated_datasets.Dataset import Dataset


class Cifar10(Dataset):
	class_names = ['plan', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

	@staticmethod
	def load_data(
			client_count: int,
			batch_size: int,
			preprocess_fn: Callable[[dict], dict],
			alpha: float = 1.,
			percent_non_iid: float = 0.,
			seed: int = 78
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
		"""
		Dataset.is_data_downloaded("cifar10")
		train_dataset, test_dataset = load_dataset(
			"cifar10",
			name="plain_text",
			cache_dir=".cache",
			split=["train", "test"],
			download_mode="reuse_dataset_if_exists"
		)
		train_dataset.set_format(type="np")
		test_dataset.set_format(type="np")

		train_dataset = train_dataset.map(preprocess_fn)
		test_dataset = test_dataset.map(preprocess_fn)

		federated_train_data, _, _, _ = SplitAsFederatedData(random_state=seed).create_clients(
			image_list=train_dataset["x"],
			label_list=train_dataset["y"],
			num_clients=client_count,
			method="dirichlet",
			alpha=alpha,
			percent_noniid=percent_non_iid
		)

		# Use with class completion so every client has at least one label of each class
		train_loaders = []
		for client_data in federated_train_data["with_class_completion"].values():
			if batch_size == -1:
				batch_size = len(client_data)
			data_loader = DataLoader(client_data, batch_size=batch_size)
			train_loaders.append(data_loader)
		test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
		return train_loaders, test_loader
