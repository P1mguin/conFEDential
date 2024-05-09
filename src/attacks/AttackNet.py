import copy
import operator
from functools import reduce

import torch
import torch.nn as nn

from src import training, utils


class AttackNet(nn.Module):
	def __init__(self, config):
		super(AttackNet, self).__init__()
		self.config = config
		self.activation_components = None
		self.label_component = None
		self.loss_component = None
		self.gradient_component = None
		self.metric_components = None
		self.encoder_component = None

		self._initialize_components()

	def forward(self, activation_values, gradients, losses, labels, metrics):
		batch_size = self.config.attack.attack_simulation.batch_size

		activation = torch.stack(
			[component(activation_value).view(batch_size, -1)
			 for component, activation_value in zip(self.activation_components, activation_values)]
		)

		label = self.label_component(labels.unsqueeze(-1)).view(batch_size, -1)
		loss = self.loss_component(losses.unsqueeze(-1)).view(batch_size, -1)

		# The input of the gradient components is
		# (attack batch size, iterations captured, out_channels, in_channels, kernel/out_features, kernel/in_features)
		# The convolution is
		gradient = torch.stack(
			[
				torch.stack([component(value).view(-1) for value in gradient_value])
				for component, gradient_value in zip(self.gradient_components, gradients)
			]
		)

		metric = torch.stack([
			torch.stack([
				torch.stack([metric_component(layer_metric_value).view(-1) for layer_metric_value in metric_value])
				for metric_component, metric_value in zip(metric_components, metric_values)
			])
			for metric_components, metric_values in zip(self.metric_components, metrics.values())
		])

		# Concatenate the activation, label, loss, gradient and metric components
		encoder_input = torch.cat([*activation, label, loss, *gradient, *metric.squeeze(0)], dim=1)
		prediction = self.encoder_component(encoder_input)
		return prediction

	def _initialize_components(self):
		self._initialize_activation_component()
		self._initialize_label_component()
		self._initialize_loss_component()
		self._initialize_gradient_component()
		self._initialize_encoder_component()
		self._initialize_metric_component()

	def _initialize_activation_component(self):
		model = self.config.simulation.model
		input_shape = self.config.simulation.data.get_input_size()
		layer_shapes = self.config.simulation.model_config.get_layer_shapes(input_shape)

		trainable_layer_indices = self.config.simulation.model_config.get_trainable_layer_indices()
		activation_components = []
		for i in range(len(model.layers)):
			# Trainable layers are accounted for in the gradients components
			if i in trainable_layer_indices:
				continue

			layer_output_shape = layer_shapes[i + 1]
			is_flattened = len(layer_output_shape) > 1
			if is_flattened:
				layer_output_shape = (reduce(operator.mul, layer_output_shape, 1),)

			activation_component = self._get_fcn_layers(layer_output_shape[0])

			if is_flattened:
				# Prepend a flattening layer to the activation component
				activation_component.insert(0, nn.Flatten(start_dim=-len(layer_shapes[i])))

			activation_component = utils.get_net_class_from_layers(activation_component)
			activation_components.append(activation_component)

		self.activation_components = [component().to(training.DEVICE) for component in activation_components]

	def _initialize_label_component(self):
		layers = self._get_fcn_layers(1)
		label_component = utils.get_net_class_from_layers(layers)
		self.label_component = label_component().to(training.DEVICE)

	def _initialize_loss_component(self):
		layers = self._get_fcn_layers(1)
		loss_component = utils.get_net_class_from_layers(layers)
		self.loss_component = loss_component().to(training.DEVICE)

	def _initialize_gradient_component(self):
		model = self.config.simulation.model
		gradient_shapes = self.config.simulation.model_config.get_gradient_shapes()

		trainable_layer_indices = self.config.simulation.model_config.get_trainable_layer_indices()
		gradient_components = []
		for i in range(len(model.layers)):
			# Non-trainable layers are accounted for in the activation components
			if i not in trainable_layer_indices:
				continue

			weights_shape, bias_shape = gradient_shapes.pop(0)

			# Based on the assumption that the input will always be 3dimensional:
			bias_shape = bias_shape + (1,)
			while len(weights_shape) < 4:
				weights_shape = (1,) + weights_shape
			while len(bias_shape) < 4:
				bias_shape = (1,) + bias_shape

			# Assume the input to be batched
			weights_shape = (1,) + weights_shape
			bias_shape = (1,) + bias_shape

			weights_convolution_component = self._get_convolution_layers(weights_shape)
			bias_convolution_component = self._get_convolution_layers(bias_shape)

			# Flatten both outputs to be a (batched) vector
			weights_convolution_component.append(nn.Flatten(1))
			bias_convolution_component.append(nn.Flatten(1))

			# Get the output channels of the last convolutional component of the attacker model
			out_channels = next(
				layer.out_channels for layer in reversed(bias_convolution_component) if hasattr(layer, "out_channels"))

			# The convolutional components will have reduced the last dimension of the gradient to 1, the rest will be
			# flattened
			weight_size = reduce(operator.mul, [out_channels, *weights_shape[2:-1], 1])
			bias_size = reduce(operator.mul, [out_channels, *bias_shape[2:-1], 1])

			# The weights are equal to (in_channels, out_channels, vertical kernel size/input features)
			# Where it is the vertical kernel size for a convolutional layer and input features for a fcl
			# Create the linear components and append
			weights_convolution_component.extend(self._get_fcn_layers(weight_size))
			bias_convolution_component.extend(self._get_fcn_layers(bias_size))

			# Flatten the output of the fcn components
			weight_net = utils.get_net_class_from_layers(weights_convolution_component)
			bias_net = utils.get_net_class_from_layers(bias_convolution_component)
			gradient_components.append((weight_net, bias_net))
		self.gradient_components = [
			component().to(training.DEVICE) for components in gradient_components for component in components
		]

	def _initialize_metric_component(self):
		# Get the metrics from the config and then copy with equal shape as the
		_, metrics = self.config.simulation.get_server_aggregates()
		self.metric_components = [
									 [copy.deepcopy(gradient_component) for gradient_component in
									  self.gradient_components]
								 ] * len(metrics)

	def _initialize_encoder_component(self):
		model = self.config.simulation.model
		# First go over the activation and gradient components
		# The gradient components will add the bias and weights in one sweep
		trainable_layer_indices = self.config.simulation.model_config.get_trainable_layer_indices()

		activation_components = copy.deepcopy(self.activation_components)
		gradient_components = copy.deepcopy(self.gradient_components)

		layer_components = activation_components + gradient_components

		global_rounds = self.config.simulation.global_rounds
		metric_count = len(self.config.simulation.get_server_aggregates()[1])

		# Get the output size of each component
		input_size = 0
		for i, layer_component in enumerate(layer_components):
			out_features = next(
				layer.out_features for layer in reversed(gradient_components[0].layers) if
				hasattr(layer, "out_features")
			)
			if i < len(activation_components):
				input_size += out_features * global_rounds
			else:
				input_size += out_features * (1 + metric_count) * global_rounds

		# Add the out features two times for the loss and label component
		input_size += out_features * 2 * global_rounds

		layers = []
		in_features_set = False
		raw_encoder_copy = copy.deepcopy(self.config.attack.attack_simulation.model_architecture.encoder)
		for layer in raw_encoder_copy:
			layer_type = getattr(nn, layer["type"])
			parameters = {key: value for key, value in list(layer.items())[1:]}

			# The first linear layer of the attack has the size of the concatenated output of all components
			if not in_features_set and layer_type is nn.Linear:
				in_features_set = True
				parameters = {**parameters, "in_features": input_size}

			layer = layer_type(**parameters)
			layers.append(layer)
		encoder_component = utils.get_net_class_from_layers(layers)
		self.encoder_component = encoder_component()

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
		return layers
		pass

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
