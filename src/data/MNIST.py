from typing import Tuple

from datasets import load_dataset

from src.datasets.Dataset import Dataset


class MNIST(Dataset):
	@staticmethod
	def load_dataset() -> Tuple[Dataset, Dataset]:
		# For documentation see Dataset
		Dataset.is_data_downloaded("mnist")
		train_dataset, test_dataset = load_dataset(
			"mnist",
			name="mnist",
			cache_dir=".cache",
			split=["train", "test"],
			download_mode="reuse_dataset_if_exists"
		)

		# Convert the pytorch tensor to NumPy such FedArtML can convert the data to non-iid
		train_dataset.set_format(type="np")
		test_dataset.set_format(type="np")

		return train_dataset, test_dataset
