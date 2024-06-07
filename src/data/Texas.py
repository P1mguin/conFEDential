from typing import Tuple

import numpy as np
from datasets import Dataset as HuggingFaceDataset, load_dataset

from src.data.Dataset import Dataset


class Texas(Dataset):
	@staticmethod
	def load_dataset() -> Tuple[HuggingFaceDataset, HuggingFaceDataset]:
		Dataset.is_data_downloaded("texas")

		# Get the file from the locally downloaded files
		dataset = load_dataset("parquet", data_files=".cache/data/texas/texas/texas.parquet")
		train_dataset = dataset["train"].select(range(10000))
		test_dataset = dataset["train"].select(range(10000, 11775))

		def split_label_and_features(entry):
			entry = np.array(list(entry.values()))
			return {
				"label": entry[0].astype(np.int64),
				"features": entry[1:].astype(np.float32)
			}

		train_dataset = train_dataset.map(split_label_and_features)
		test_dataset = test_dataset.map(split_label_and_features)

		return train_dataset, test_dataset
