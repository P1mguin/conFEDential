from typing import Callable, List, Tuple

from datasets import load_dataset
from fedartml import SplitAsFederatedData
from torch.utils.data import DataLoader

from .Dataset import Dataset


class MNIST(Dataset):
	@staticmethod
	def load_data(client_count: int, batch_size: int, preprocess_fn: Callable[[dict], dict], alpha: float = 1.,
				  percent_noniid: float = 0., seed: int = 78) -> Tuple[List[DataLoader], DataLoader]:
		train_dataset, test_dataset = load_dataset("mnist", cache_dir="cache/mnist", split=["train", "test"])
		train_dataset.set_format(type="np")
		test_dataset.set_format(type="np")

		train_dataset = train_dataset.map(preprocess_fn)
		test_dataset = test_dataset.map(preprocess_fn)

		federated_train_data, _, _, _ = SplitAsFederatedData(random_state=seed).create_clients(
			image_list=train_dataset["x"], label_list=train_dataset["y"], num_clients=client_count, method="dirichlet",
			alpha=alpha, percent_noniid=percent_noniid)

		trainloaders = []
		for client_data in federated_train_data["with_class_completion"].values():
			if batch_size == -1:
				batch_size = len(client_data)
			dataloader = DataLoader(client_data, batch_size=batch_size)
			trainloaders.append(dataloader)
		testloader = DataLoader(test_dataset, batch_size=len(test_dataset))
		return trainloaders, testloader
