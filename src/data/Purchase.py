from typing import Tuple

import numpy as np
import torch
from torch.utils.data import random_split, TensorDataset

from src.datasets.Dataset import Dataset


class Purchase(Dataset):
	@staticmethod
	def load_dataset() -> Tuple[Dataset, Dataset]:
		# For documentation see Dataset
		Dataset.is_data_downloaded("purchase")

		purchase = np.load(".cache/purchase/purchase/purchase100.npz")
		features = torch.from_numpy(purchase["features"])
		labels = torch.from_numpy(np.argmax(purchase["labels"], axis=1))

		# Split the features and labels into a train and test set in the same way
		dataset = TensorDataset(features, labels)
		train_size = int(0.85 * len(dataset))
		test_size = len(dataset) - train_size
		train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

		return train_dataset, test_dataset
