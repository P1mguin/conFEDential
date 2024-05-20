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
		train_dataset = load_dataset("csv", data_files=".cache/data/texas/texas/texas100.npz", split="train[:85%]")
		test_dataset = load_dataset("csv", data_files=".cache/data/texas/texas/texas100.npz", split="train[-15%:]")

		def convert_to_array_and_int64(entry):
			return {
				"label": entry["label"],
				"features": np.array(entry["features"].strip("[]").split()).astype(np.float64)
			}

		train_dataset = train_dataset.map(convert_to_array_and_int64)
		test_dataset = test_dataset.map(convert_to_array_and_int64)

		return train_dataset, test_dataset
