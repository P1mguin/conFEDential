from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset

from src.data.Dataset import Dataset


class Texas(Dataset):
	@staticmethod
	def load_dataset() -> Tuple[DataLoader, DataLoader]:
		Dataset.is_data_downloaded("texas")

		texas = np.load(".cache/data/texas/texas/texas100.npz")
		features = torch.from_numpy(texas["features"])
		labels = torch.from_numpy(np.argmax(texas["labels"], axis=1))

		# Split the features and labels into a train and test set in the same way
		dataset = TensorDataset(features, labels)
		train_size = int(0.85 * len(dataset))
		test_size = len(dataset) - train_size
		train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

		return train_dataset, test_dataset
