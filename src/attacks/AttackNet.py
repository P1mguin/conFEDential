import copy
import operator
from functools import reduce

import torch
import torch.nn as nn

from src import training, utils


class AttackNet(nn.Module):
	def __init__(self, config, gradient_shapes, activation_shapes, metrics_shapes, label_shape):
		super(AttackNet, self).__init__()
		self.config = config

		self.label_component = None
		self.loss_component = None
		self.metric_components = None
		self.activation_components = None
		self.gradient_component = None
		self.encoder_component = None

		self._initialize_components(gradient_shapes, activation_shapes, metrics_shapes, label_shape)

	def forward(self, gradients, activation_values, metrics, loss_value, label):
		activation = torch.cat([
			activation_component(activation_value)
			for activation_component, activation_value in zip(self.activation_components, activation_values)
		], dim=1)

		label = self.label_component(label)
		loss = self.loss_component(loss_value)

		# The input of the gradient components is
		# (attack batch size, iterations captured, out_channels, in_channels, kernel/out_features, kernel/in_features)
		# The convolution is
		gradient = torch.cat([
			torch.stack([
				gradient_component(gradient).view(-1) for gradient in gradient_iterations
			]) for gradient_iterations, gradient_component in zip(gradients, self.gradient_components)
		], dim=1)

		metric = torch.cat([
			torch.cat([
				torch.stack([metric_component(metric).view(-1) for metric in metric_iterations])
				for metric_iterations, metric_component in zip(metrics[key], self.metric_components[key])
			], dim=1) for key in metrics.keys()
		], dim=1)

		# Concatenate the activation, label, loss, gradient and metric components
		encoder_input = torch.cat((label, loss, metric, activation, gradient), dim=1)
		prediction = self.encoder_component(encoder_input).view(-1)
		return prediction

	def _initialize_components(self, gradient_shapes, activation_shapes, metrics_shapes, label_shape):
		self._initialize_label_component(label_shape)
		self._initialize_loss_component()
		self._initialize_metric_component(metrics_shapes)
		self._initialize_activation_component(activation_shapes)
		self._initialize_gradient_component(gradient_shapes)
		self._initialize_encoder_component()

	def _initialize_activation_component(self, activation_shapes):
		activation_components = [[nn.Flatten(start_dim=2)] for _ in activation_shapes]
		activation_shapes = [shape[1:] for shape in activation_shapes]
		activation_shapes = [reduce(operator.mul, shape, 1) for shape in activation_shapes]

		for i, activation_shape in enumerate(activation_shapes):
			activation_component = self._get_fcn_layers(activation_shape)
			activation_components[i].extend(activation_component)
			activation_components[i] = utils.get_net_class_from_layers(activation_components[i])

		self.activation_components = [component().to(training.DEVICE) for component in activation_components]

	def _initialize_label_component(self, label_shape):
		layers = self._get_fcn_layers(label_shape[0])
		label_component = utils.get_net_class_from_layers(layers)
		self.label_component = label_component().to(training.DEVICE)

	def _initialize_loss_component(self):
		layers = self._get_fcn_layers(1)
		loss_component = utils.get_net_class_from_layers(layers)
		self.loss_component = loss_component().to(training.DEVICE)

	def _initialize_gradient_component(self, gradient_shapes):
		gradient_components = [self._get_convolution_layers(gradient_shape) for gradient_shape in gradient_shapes]

		# Flatten the output of the convolutional components
		for gradient_component in gradient_components:
			gradient_component.append(nn.Flatten(1))

		# The out channels is a result from the config, so is the same for each gradient
		out_channels = next(
			layer.out_channels for layer in reversed(gradient_components[0]) if hasattr(layer, "out_channels")
		)

		# Compute the output size of the convolutional components
		output_sizes = [(out_channels, *gradient_shape[2:-1], 1) for gradient_shape in gradient_shapes]
		output_sizes = [reduce(operator.mul, output_size, 1) for output_size in output_sizes]

		# Add the fcn components to the convolutional components
		for gradient_component, output_size in zip(gradient_components, output_sizes):
			gradient_component.extend(self._get_fcn_layers(output_size))

		# Make the gradient component
		gradient_components = [
			utils.get_net_class_from_layers(gradient_component) for gradient_component in gradient_components
		]
		self.gradient_components = [component().to(training.DEVICE) for component in gradient_components]

	def _initialize_metric_component(self, metrics_shapes):
		# Get the metrics from the config and then copy with equal shape as the
		self.metric_components = {}
		for key, metric_shapes in metrics_shapes.items():
			metric_components = [self._get_convolution_layers(metric_shape) for metric_shape in metric_shapes]

			# Flatten the output of the convolutional components
			for metric_component in metric_components:
				metric_component.append(nn.Flatten(1))

			# The out channels is a result from the config, so is the same for each metric
			out_channels = next(
				layer.out_channels for layer in reversed(metric_components[0]) if hasattr(layer, "out_channels")
			)

			# Compute the output size of the convolutional components
			output_sizes = [(out_channels, *metric_shape[2:-1], 1) for metric_shape in metric_shapes]
			output_sizes = [reduce(operator.mul, output_size, 1) for output_size in output_sizes]

			# Add the fcn components to the convolutional components
			for metric_component, output_size in zip(metric_components, output_sizes):
				metric_component.extend(self._get_fcn_layers(output_size))

			# Make the metric component
			metric_components = [
				utils.get_net_class_from_layers(metric_component) for metric_component in metric_components
			]
			self.metric_components[key] = [component().to(training.DEVICE) for component in metric_components]

	def _initialize_encoder_component(self):
		global_rounds = self.config.simulation.global_rounds
		concatenated_size = 0
		get_output_size = lambda component: next(
			layer.out_features for layer in reversed(component.layers) if hasattr(layer, "out_features")
		)
		concatenated_size += get_output_size(self.label_component)
		concatenated_size += get_output_size(self.loss_component) * (global_rounds + 1)
		for metric_components in self.metric_components.values():
			for metric_component in metric_components:
				concatenated_size += get_output_size(metric_component) * (global_rounds + 1)
		for activation_component in self.activation_components:
			concatenated_size += get_output_size(activation_component) * (global_rounds + 1)
		for gradient_component in self.gradient_components:
			concatenated_size += get_output_size(gradient_component) * (global_rounds + 1)

		layers = []
		in_features_set = False
		raw_encoder_copy = copy.deepcopy(self.config.attack.attack_simulation.model_architecture.encoder)
		for layer in raw_encoder_copy:
			layer_type = getattr(nn, layer["type"])
			parameters = {key: value for key, value in list(layer.items())[1:]}

			# The first linear layer of the attack has the size of the concatenated output of all components
			if not in_features_set and layer_type is nn.Linear:
				in_features_set = True
				parameters = {**parameters, "in_features": concatenated_size}

			layer = layer_type(**parameters)
			layers.append(layer)
		encoder_component = utils.get_net_class_from_layers(layers)
		self.encoder_component = encoder_component().to(training.DEVICE)

	def _get_fcn_layers(self, input_size: int):
		layers = []
		raw_fcn_copy = copy.deepcopy(self.config.attack.attack_simulation.model_architecture.fcn)
		dynamic_parameters_set = False
		for layer in raw_fcn_copy:
			layer_type = getattr(nn, layer["type"])
			parameters = {key: value for key, value in list(layer.items())[1:]}

			# The first linear layer of the attack has as many inputs as there are output classes
			if not dynamic_parameters_set and layer_type is nn.Linear:
				dynamic_parameters_set = True
				parameters = {**parameters, "in_features": input_size}

			layer = layer_type(**parameters)
			layers.append(layer)
		layers.append(nn.Flatten(start_dim=1))
		return layers

	def _get_convolution_layers(self, input_shape):
		layers = []
		raw_cnn_copy = copy.deepcopy(self.config.attack.attack_simulation.model_architecture.gradient)
		dynamic_parameters_set = False
		for layer in raw_cnn_copy:
			# Get the static parameters
			if hasattr(nn, layer["type"]):
				module = nn
			else:
				module = utils

			layer_type = getattr(module, layer["type"])
			parameters = {key: value for key, value in list(layer.items())[1:]}

			# The kernel of the first convolution layer is as wide as the width/breadth of the input shape
			if not dynamic_parameters_set and (layer_type is nn.Conv2d or layer_type is nn.Conv3d):
				parameters["in_channels"] = input_shape[1]
				parameters["kernel_size"].append(input_shape[-1])
				dynamic_parameters_set = True

			layer = layer_type(**parameters)
			layers.append(layer)
		return layers
