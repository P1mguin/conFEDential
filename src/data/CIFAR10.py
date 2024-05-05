from typing import List, Tuple

from datasets import load_dataset
from torch.utils.data import DataLoader

from src.datasets.Dataset import Dataset


class CIFAR10(Dataset):
	@staticmethod
	def load_dataset() -> Tuple[List[DataLoader], DataLoader]:
		# For documentation see Dataset
		Dataset.is_data_downloaded("cifar10")
		train_dataset, test_dataset = load_dataset(
			"cifar10",
			name="plain_text",
			cache_dir=".cache",
			split=["train", "test"],
			download_mode="reuse_dataset_if_exists"
		)

		# Convert the pytorch tensor to NumPy such FedArtML can convert the data to non-iid
		train_dataset.set_format(type="np")
		test_dataset.set_format(type="np")

		return train_dataset, test_dataset
