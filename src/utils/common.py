from functools import reduce
from typing import Any, Callable, Iterator, List, Tuple, Type

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, random_split


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
	if kernel_size is int:
		kernel_size = (kernel_size,) * len(input_size)

	if stride is int:
		stride = (stride,) * len(input_size)

	if padding is int:
		padding = (padding,) * len(input_size)

	# Compute the output size for each dimension separately
	output_shape = []
	for i in range(len(input_size)):
		output_size = ((input_size[i] - kernel_size[i] + 2 * padding[i]) // stride[i]) + 1
		output_shape.append(output_size)

	return out_channels, *output_shape


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


def get_model_layer_shapes(model: nn.Module, run_config) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
	"""
	Returns the input and output shape of each layer in the model
	:param model: the model to get the layer shapes from
	:param run_config: the configuration of the experiment
	"""
	sizes = []
	input_shape, output_shape = None, None
	for layer in model.layers:
		# The input size of the layer is either indicated by the layer itself,
		# or it is the input size of the previous layer. If no previous layer has indicated a previous shape,
		# the data is not transformed and so it is the output shape of the previous layer
		if hasattr(layer, "in_features"):
			input_shape = (layer.in_features,)
		elif input_shape is None:
			input_shape = get_config_input_shape(run_config)
		else:
			input_shape = output_shape

		# The output size of the layer is either indicated by the layer itself,
		# or the layer does not change the shape of the data, which makes it equal to the input shape of the layer
		if hasattr(layer, "out_features"):
			output_shape = (layer.out_features,)
		else:
			output_shape = input_shape

		# TODO: Fix size of convolutional components

		sizes.append((input_shape, output_shape))
	return sizes


def get_config_input_shape(run_config) -> Tuple[int, ...]:
	"""
	Returns the input shape of the model from the configuration
	:param run_config: the configuration for which the experiment is run
	"""
	_, test_loader = run_config.get_dataloaders()
	input_shape = tuple(next(iter(test_loader))["x"].shape[1:])
	return input_shape

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
