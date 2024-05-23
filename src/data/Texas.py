from typing import Tuple

import numpy as np
import torch
from datasets import Dataset as HuggingFaceDataset, load_dataset

from src.data.Dataset import Dataset


class Texas(Dataset):
	@staticmethod
	def load_dataset() -> Tuple[HuggingFaceDataset, HuggingFaceDataset]:
		Dataset.is_data_downloaded("texas")

		# Get the file from the locally downloaded files
		train_dataset = load_dataset("csv", data_files=".cache/data/texas/texas/texas100.csv", split="train[:85%]")
		test_dataset = load_dataset("csv", data_files=".cache/data/texas/texas/texas100.csv", split="train[-15%:]")

		train_dataset.set_format(type="np")
		test_dataset.set_format(type="np")

		def split_label_and_features(entry):
			entry = np.array(list(entry.values()))
			return {
				"label": entry[0],
				"features": entry[1:]
			}

		train_dataset = train_dataset.map(split_label_and_features)
		test_dataset = test_dataset.map(split_label_and_features)

		return train_dataset, test_dataset
