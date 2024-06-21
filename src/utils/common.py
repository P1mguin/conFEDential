from typing import Any, List, Tuple, Type

import torch
import torch.nn as nn


def compute_weighted_average(values, counts) -> Any:
	"""
	Computes the weighted average of a generator of tuple with the first item a list of numpy arrays and the second item
	the weight of the list
	"""
	total = sum(counts)
	multiplied_values = ((layer * weight / total for layer in value) for value, weight in zip(values, counts))

	for layer in zip(*multiplied_values):
		yield sum(layer)


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


def get_net_class_from_layers(layers: List[Type[nn.Module]]) -> Type[nn.Module]:
	"""
	Constructs a model class from a list of layers
	:param layers: the layers that should be included in the model
	"""

	class Net(nn.Module):
		def __init__(self) -> None:
			super(Net, self).__init__()
			self.layers = nn.Sequential(*layers)

		def forward(self, x: torch.Tensor) -> torch.Tensor:
			x = self.layers(x)
			return x

	return Net


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
