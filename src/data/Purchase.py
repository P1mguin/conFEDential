from typing import Tuple

import numpy as np
import torch
from datasets import Dataset as HuggingFaceDataset, load_dataset

from src.data.Dataset import Dataset


class Purchase(Dataset):
	@staticmethod
	def load_dataset() -> Tuple[HuggingFaceDataset, HuggingFaceDataset]:
		Dataset.is_data_downloaded("purchase")

		# Get the file from the locally downloaded files
		train_dataset = load_dataset(
			"csv", data_files=".cache/data/purchase/purchase/purchase100.csv", split="train[:85%]"
		)
		test_dataset = load_dataset(
			"csv", data_files=".cache/data/purchase/purchase/purchase100.csv", split="train[-15%:]"
		)

		def split_label_and_features(entry):
			entry = np.array(list(entry.values()))
			return {
				"label": entry[0],
				"features": entry[1:]
			}

		train_dataset = train_dataset.map(split_label_and_features)
		test_dataset = test_dataset.map(split_label_and_features)

		return train_dataset, test_dataset
