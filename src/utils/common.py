from functools import reduce
from typing import Any, Callable, Iterator, List, Tuple

import numpy as np
import numpy.typing as npt
from torch.utils.data import ConcatDataset, DataLoader, random_split


def compute_weighted_average(values: List[Tuple[List[npt.NDArray], int]]) -> Any:
	"""
	Computes the weighted average of a list of tuple with the first item a list of numpy arrays and the second item
	the weight of the list
	:param values: the list of tuples
	"""
	total_count = sum(num_examples for (_, num_examples) in values)
	values = [[layer * num_examples for layer in weights] for weights, num_examples in values]
	average = [reduce(np.add, layers) / total_count for layers in zip(*values)]
	return average


def find_all_paths(dictionary: dict, key: str, path=None):
	"""
	Recursive function that finds all path to a given key name in a dictionary
	:param dictionary: the dictionary to search
	:param key: the key to which the paths need to be found
	:param path: A helper variable for the recursion
	"""
	if path is None:
		path = []

	paths = []
	if isinstance(dictionary, dict):
		for k, v in dictionary.items():
			new_path = path + [k]
			if k == key:
				paths.append(new_path)
			else:
				paths.extend(find_all_paths(v, key, new_path))
	return paths


def get_dict_value_from_path(dictionary: dict, *path: Tuple[str]) -> Any:
	"""
	Common function that gets a value from a dict given a path
	:param dictionary: the dictionary to get value from
	:param path: the path of the value
	"""
	value = dictionary
	for key in path:
		value = value[key]
	return value


def k_fold_dataset(dataset, k: int, batch_size: int) -> Iterator[Tuple[DataLoader, DataLoader]]:
	"""
	Combines a dataset into a list of train and validation loaders by k-folding the data
	:param dataset: the dataset to k-fold
	:param k: the amount of splits that should be made, the size of the generator will be equal to k
	:param batch_size: the batch size of the dataloaders
	"""
	fold_size = len(dataset) // k

	# Create k subsets
	subsets = random_split(dataset, [fold_size] * (k - 1) + [len(dataset) - fold_size * (k - 1)])

	# For each fold, create a DataLoader for the training set and the validation set
	for i in range(k):
		validation_set = subsets[i]
		training_sets = subsets[:i] + subsets[i + 1:]
		training_set = ConcatDataset(training_sets)

		yield (DataLoader(training_set, batch_size=batch_size, shuffle=True),
			   DataLoader(validation_set, batch_size=batch_size))

def load_func_from_function_string(function_string: str, function_name: str) -> Callable[[Any], Any]:
	"""
	Loads and parses a python function described as string
	:param function_string: python function with correct indentation
	:param function_name: the name of the function in the string
	"""
	namespace = {}
	exec(function_string, namespace)
	return namespace[function_name]


def set_dict_value_from_path(dictionary: dict, new_value: Any, *path: Tuple[str]) -> dict:
	"""
	Common function that sets a value from a dict given a path, sets it in place but also returns the part
	of the dictionary that was updated
	:param dictionary: the dictionary on which the value should be set
	:param new_value: the new value to set
	:param path: the path at which the new value should be set
	"""
	for key in path[:-1]:
		dictionary = dictionary[key]

	dictionary[path[-1]] = new_value
	return dictionary
