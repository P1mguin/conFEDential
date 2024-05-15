from typing import Any, Callable, List, Set, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_convolution_output_size(
		input_size: Tuple[int, ...],
		out_channels: int,
		kernel_size: Tuple[int, int] | int,
		stride: Tuple[int, int] | int = 1,
		padding: Tuple[int, int] | int = 0,
) -> Tuple[int, ...]:
	"""
	Computes the output size of a convolutional layer given the input size and the parameters of the convolutional layer
	:param input_size: the input size that is expected of the convolution layer
	:param out_channels: the amount of out channels
	:param kernel_size: the size of the kernel
	:param stride: the step size of the kernel
	:param padding: the size of the padding
	"""
	# Convert all ints to tuples
	if isinstance(kernel_size, int):
		kernel_size = (kernel_size,) * len(input_size)

	if isinstance(stride, int):
		stride = (stride,) * len(input_size)

	if isinstance(padding, int):
		padding = (padding,) * len(input_size)

	# Compute the output size for each dimension separately
	output_shape = []
	for i in range(len(input_size)):
		output_size = ((input_size[i] - kernel_size[i] + 2 * padding[i]) // stride[i]) + 1
		output_shape.append(output_size)

	return out_channels, *output_shape


def compute_weighted_average(values, counts) -> Any:
	"""
	Computes the weighted average of a generator of tuple with the first item a list of numpy arrays and the second item
	the weight of the list
	"""
	total = sum(counts)
	multiplied_values = ((layer * weight / total for layer in value) for value, weight in zip(values, counts))
	mean_values = (np.sum(layer, axis=0) for layer in zip(*multiplied_values))
	return mean_values


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


def get_trainable_layers_indices(model: nn.Module) -> Set[int]:
	"""
	Returns the indices of the trainable layers
	:param model: the model to get the trainable layers from
	"""
	return set(int(name.split(".")[1]) for name, param in model.named_parameters() if param.requires_grad)


def load_func_from_function_string(function_string: str, function_name: str) -> Callable[[Any], Any]:
	"""
	Loads and parses a python function described as string
	:param function_string: python function with correct indentation
	:param function_name: the name of the function in the string
	"""
	namespace = {}
	exec(function_string, namespace)
	return namespace[function_name]


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


def split_dataloader(dataloader: DataLoader, percentage: float) -> Tuple[DataLoader, DataLoader]:
	"""
	Splits a dataloader into two dataloaders based on a given percentage
	:param dataloader: the dataloder to split
	:param percentage: the percentage the first dataloader will receive
	"""
	dataset = dataloader.dataset
	batch_size = dataloader.batch_size
	total_length = len(dataset)
	first_length = int(total_length * percentage)
	second_length = total_length - first_length

	first_dataset, second_dataset = torch.utils.data.random_split(dataset, [first_length, second_length])
	first_loader = DataLoader(first_dataset, batch_size=batch_size, shuffle=True)
	second_loader = DataLoader(second_dataset, batch_size=batch_size, shuffle=True)
	return first_loader, second_loader
