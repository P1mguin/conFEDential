from typing import Tuple

from datasets import Dataset as HuggingFaceDataset, load_dataset, concatenate_datasets
from torchvision import transforms

from src.data.Dataset import Dataset


class CIFAR10(Dataset):
	@staticmethod
	def load_dataset(cache_root: str) -> Tuple[HuggingFaceDataset, HuggingFaceDataset, HuggingFaceDataset]:
		# For documentation see Dataset
		Dataset.is_data_downloaded("cifar10", cache_root)
		datasets = load_dataset(
			"cifar10",
			name="plain_text",
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

		train_transformation = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.RandomRotation(15),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

		test_transformations = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

		def train_augmentation(x):
			x["img"] = train_transformation(x["img"])
			return x

		def test_augmentation(x):
			x["img"] = test_transformations(x["img"])
			return x

		train_dataset = train_dataset.map(train_augmentation)
		test_dataset = test_dataset.map(test_augmentation)
		non_member_dataset = non_member_dataset.map(test_augmentation)

		# Convert the pytorch tensor to NumPy such FedArtML can convert the data to non-iid
		train_dataset.set_format(type="np")
		test_dataset.set_format(type="np")
		non_member_dataset.set_format(type="np")

		return train_dataset, test_dataset, non_member_dataset
