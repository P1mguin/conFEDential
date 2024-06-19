from typing import Tuple

import numpy as np
from datasets import Dataset as HuggingFaceDataset, load_dataset

from src.data.Dataset import Dataset


class Texas(Dataset):
	@staticmethod
	def load_dataset(cache_root: str) -> Tuple[HuggingFaceDataset, HuggingFaceDataset, HuggingFaceDataset]:
		Dataset.is_data_downloaded("texas", cache_root)

		# Get the file from the locally downloaded files
		dataset = load_dataset("parquet", data_files=f"{cache_root}data/texas/texas/texas.parquet")

		train_size = 10000
		test_size = 2500
		non_member_size = 7500

		train_dataset = dataset["train"].select(range(train_size))
		test_dataset = dataset["train"].select(range(train_size, train_size + test_size))
		non_member_size = dataset["train"].select(
			range(train_size + test_size, train_size + test_size + non_member_size)
		)

		def split_label_and_features(entry):
			entry = np.array(list(entry.values()))
			return {
				"label": entry[0].astype(np.int64),
				"features": entry[1:].astype(np.float32)
			}

		train_dataset = train_dataset.map(split_label_and_features)
		test_dataset = test_dataset.map(split_label_and_features)
		non_member_size = non_member_size.map(split_label_and_features)

		return train_dataset, test_dataset, non_member_size
