from typing import Tuple

from datasets import load_dataset

from src.datasets.Dataset import Dataset


class ImageNet(Dataset):
	@staticmethod
	def load_dataset() -> Tuple[Dataset, Dataset]:
		# For documentation see Dataset
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

		return train_dataset, test_dataset
