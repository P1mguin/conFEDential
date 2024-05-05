from typing import Tuple

from datasets import load_dataset

from src.datasets.Dataset import Dataset


class CIFAR100(Dataset):
	@staticmethod
	def load_dataset() -> Tuple[Dataset, Dataset]:
		# For documentation see Dataset
		Dataset.is_data_downloaded("cifar100")
		train_dataset, test_dataset = load_dataset(
			"cifar100",
			name="cifar100",
			cache_dir=".cache",
			split=["train", "test"],
			download_mode="reuse_dataset_if_exists"
		)

		# Convert the pytorch tensor to NumPy such FedArtML can convert the data to non-iid
		train_dataset.set_format(type="np")
		test_dataset.set_format(type="np")

		return train_dataset, test_dataset
