import copy
import operator
from functools import reduce
from typing import Dict, List, Sequence

import torch.nn as nn


def construct_label_component(config, label_shape: Sequence[int]) -> nn.Module:
	"""
	Constructs the component that is used in the attacks to model the distribution
	and importance of the one-hot-encoded target label of the input data
	:param config: the configuration with which the experiment is run
	:param label_shape: the shape of the one-hot-encoded target label
	"""
	layers = _get_fcn_layers(config, label_shape[0])
	return nn.Sequential(*layers)


def construct_loss_component(config) -> nn.Module:
	"""
	Constructs the component that is used in the attacks to model the distribution
	and importance of the loss values of the victim model
	:param config: the configuration with which the experiment is run
	"""
	layers = _get_fcn_layers(config, 1)
	layers.append(nn.Flatten(start_dim=1))
	return nn.Sequential(*layers)


def construct_activation_component(config, activation_shapes: List[Sequence[int]]) -> nn.Module:
	"""
	Constructs the component that is used in the attacks to model the distribution
	and importance of each element in the activation values of the victim model
	:param config: the configuration with which the experiment is run
	:param activation_shapes: the output shapes of each layer of the model
	"""
	# Flatten the activation components along the second axis, the first is the global rounds. They go into an
	# FCN so need flattening
	activation_components = [[nn.Flatten(start_dim=2)] for _ in activation_shapes]
	activation_shapes = [shape[1:] for shape in activation_shapes]
	activation_shapes = [reduce(operator.mul, shape, 1) for shape in activation_shapes]

	for activation_shape, activation_component in zip(activation_shapes, activation_components):
		activation_component.extend(_get_fcn_layers(config, activation_shape))

	activation_components = nn.ModuleList([nn.Sequential(*component) for component in activation_components])
	return activation_components


def construct_gradient_component(config, gradient_shapes: List[Sequence[int]]) -> nn.Module:
	"""
	Constructs the component that is used in the attacks to model the distribution
	and importance of each element in the gradient values of the victim model
	:param config: the configuration with which the experiment is run
	:param gradient_shapes: the output shapes of the trainable layers of the model, i.e. those
	with a gradient
	"""
	gradient_components = [_get_convolution_layers(config, gradient_shape) for gradient_shape in gradient_shapes]

	out_channels = next(
		layer.out_channels for layer in reversed(gradient_components[0]) if hasattr(layer, "out_channels")
	)

	# Add the fcn components to the convolutional components
	for gradient_component in gradient_components:
		gradient_component.append(nn.Flatten(start_dim=-3))
		gradient_component.extend(_get_fcn_layers(config, out_channels))

	gradient_components = nn.ModuleList([nn.Sequential(*component) for component in gradient_components])
	return gradient_components


def construct_metric_component(config, metrics_shapes: Dict[str, List[Sequence[int]]]) -> nn.Module:
	"""
	Constructs the component that is used in the attacks to model the distribution
	and importance of each additional piece of information and its elements that the federation
	protocol requires each client to send to the server and vice-vers.a
	:param config: the configuration with which the experiment is run
	:param metrics_shapes: the output shapes of the additional pieces of information
	"""
	metric_component_dict = {}
	for key, metric_shapes in metrics_shapes.items():
		metric_components = [_get_convolution_layers(config, metric_shape) for metric_shape in metric_shapes]

		out_channels = next(
			layer.out_channels for layer in reversed(metric_components[0]) if hasattr(layer, "out_channels")
		)

		# Add the fcn components to the convolutional components
		for metric_component in metric_components:
			metric_component.append(nn.Flatten(start_dim=-3))
			metric_component.extend(_get_fcn_layers(config, out_channels))

		metric_component_dict[key] = nn.ModuleList([nn.Sequential(*component) for component in metric_components])
	metric_component_dict = nn.ModuleDict(metric_component_dict)
	return metric_component_dict


def construct_encoder_component(config, input_size: int, output_size: int) -> nn.Module:
	"""
	Constructs the component that bundles the outputs of each component to determine the
	distribution and importance of each of these variables.
	:param config: the configuration with which the experiment is run
	:param input_size: the size of the concatenated inputs
	:param output_size: the size the output layer should be, e.g. 1 in case of membership inference
	"""
	# Construct the encoder component knowing the size of the input
	layers = []
	raw_encoder = copy.deepcopy(config.attack.attack_simulation.model_architecture.encoder)
	in_features_set = False
	for layer in raw_encoder:
		layer_type = getattr(nn, layer["type"])
		parameters = {key: value for key, value in list(layer.items())[1:]}

		# The first linear layer of the attack has the size of the concatenated output of all components
		if not in_features_set and layer_type is nn.Linear:
			in_features_set = True
			parameters = {**parameters, "in_features": input_size}

		layer = layer_type(**parameters)
		layers.append(layer)

	# Add a final linear layer to get to the output size
	output_layer_input_size = next(layer.out_features for layer in reversed(layers) if hasattr(layer, "out_features"))
	layers.append(nn.Linear(in_features=output_layer_input_size, out_features=output_size))

	encoder_component = nn.Sequential(*layers)
	return encoder_component


def _get_fcn_layers(config, input_size: int) -> List[nn.Module]:
	"""
	Returns the fully connected subcomponent of the variable component.
	:param config: the configuration with which the experiment is run
	:param input_size: the size of the input layer
	"""
	layers = []
	raw_fcn = copy.deepcopy(config.attack.attack_simulation.model_architecture.fcn)
	dynamic_parameters_set = False
	for layer in raw_fcn:
		layer_type = getattr(nn, layer["type"])
		parameters = {key: value for key, value in list(layer.items())[1:]}

		# The first linear layer of the attack has as many inputs as there are output classes
		if not dynamic_parameters_set and layer_type is nn.Linear:
			dynamic_parameters_set = True
			parameters = {**parameters, "in_features": input_size}

		layer = layer_type(**parameters)
		layers.append(layer)
	return layers


def _get_convolution_layers(config, input_shape: Sequence[int]) -> List[nn.Module]:
	"""
	Returns the convolutional subcomponent of the variable component.
	:param config: the configuration with which the experiment is run
	:param input_shape: the shape of the input grid
	"""
	layers = []
	raw_cnn = copy.deepcopy(config.attack.attack_simulation.model_architecture.gradient)
	dynamic_parameters_set = False
	for layer in raw_cnn:
		layer_type = getattr(nn, layer["type"])
		parameters = {key: value for key, value in list(layer.items())[1:]}

		# The kernel of the first convolution layer is as wide as the width/breadth of the input shape
		if not dynamic_parameters_set and layer_type is nn.Conv2d:
			parameters["in_channels"] = input_shape[-3]
			parameters["kernel_size"] = input_shape[-2:]
			dynamic_parameters_set = True

		layer = layer_type(**parameters)
		layers.append(layer)
	return layers
