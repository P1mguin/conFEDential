from typing import Tuple

from datasets import Dataset as HuggingFaceDataset, load_dataset, concatenate_datasets

from src.data.Dataset import Dataset


class MNIST(Dataset):
	@staticmethod
	def load_dataset(cache_root: str) -> Tuple[HuggingFaceDataset, HuggingFaceDataset, HuggingFaceDataset]:
		# For documentation see Dataset
		Dataset.is_data_downloaded("mnist", cache_root)
		datasets = load_dataset(
			"mnist",
			name="mnist",
			cache_dir=f"{cache_root}data",
			split=["train", "test"],
			download_mode="reuse_dataset_if_exists"
		)

		dataset = concatenate_datasets(datasets)
		train_size = int(dataset.shape[0] * 0.5)
		test_size = int(dataset.shape[0] * 0.125)
		non_member_size = int(dataset.shape[0] * 0.375)

		train_dataset = dataset.select(range(train_size))
		test_dataset = dataset.select(range(train_size, train_size + test_size))
		non_member_dataset = dataset.select(range(train_size + test_size, train_size + test_size + non_member_size))

		# Convert the pytorch tensor to NumPy such FedArtML can convert the data to non-iid
		train_dataset.set_format(type="np")
		test_dataset.set_format(type="np")
		non_member_dataset.set_format(type="np")

		return train_dataset, test_dataset, non_member_dataset
