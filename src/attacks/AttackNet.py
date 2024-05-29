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
		label = self.label_component(label)
		loss = self.loss_component(loss_value)
		activation = torch.cat([
			activation_component(activation_value)
			for activation_component, activation_value in zip(self.activation_components, activation_values)
		], dim=1)

		# The input shape of the gradients and metrics is:
		# (in_channels/in_features, out_channels/1, kernel_width/1, kernel_height/out_features)
		batch_size = self.config.attack.attack_simulation.batch_size
		gradients = [gradient.view(-1, *gradient.shape[-3:]) for gradient in gradients]
		gradients = torch.cat([
			gradient_component(gradient).view(batch_size, -1)
			for gradient_component, gradient in zip(self.gradient_components, gradients)
		], dim=1)
		metrics = {
			key: [value.view(-1, *value.shape[-3:]) for value in metrics[key]]
			for key in metrics.keys()
		}
		metrics = [
			torch.cat([
				metric_component(metric_value).view(batch_size, -1)
				for metric_component, metric_value in zip(self.metric_components[key], metrics[key])
			], dim=1) for key in metrics.keys()
		]

		# Concatenate the activation, label, loss, gradient and metric components
		encoder_input = torch.cat((label, loss, activation, gradients, *metrics), dim=1)
		prediction = self.encoder_component(encoder_input).squeeze(1)
		return prediction

	def _initialize_components(self, gradient_shapes, activation_shapes, metrics_shapes, label_shape):
		self._initialize_label_component(label_shape)
		self._initialize_loss_component()
		self._initialize_activation_component(activation_shapes)
		self._initialize_gradient_component(gradient_shapes)
		self._initialize_metric_component(metrics_shapes)
		self._initialize_encoder_component(activation_shapes, gradient_shapes, metrics_shapes)

	def _initialize_label_component(self, label_shape):
		layers = self._get_fcn_layers(label_shape[0])
		label_component = utils.get_net_class_from_layers(layers)
		self.label_component = label_component().to(training.DEVICE)

	def _initialize_loss_component(self):
		layers = self._get_fcn_layers(1)
		layers.append(nn.Flatten(start_dim=1))
		loss_component = utils.get_net_class_from_layers(layers)
		self.loss_component = loss_component().to(training.DEVICE)

	def _initialize_activation_component(self, activation_shapes):
		activation_components = [[nn.Flatten(start_dim=2)] for _ in activation_shapes]
		activation_shapes = [shape[1:] for shape in activation_shapes]
		activation_shapes = [reduce(operator.mul, shape, 1) for shape in activation_shapes]

		for i, activation_shape in enumerate(activation_shapes):
			activation_component = self._get_fcn_layers(activation_shape)
			activation_components[i].extend(activation_component)
			activation_components[i].append(nn.Flatten(start_dim=1))
			activation_components[i] = utils.get_net_class_from_layers(activation_components[i])

		self.activation_components = [component().to(training.DEVICE) for component in activation_components]

	def _initialize_gradient_component(self, gradient_shapes):
		gradient_components = [self._get_convolution_layers(gradient_shape) for gradient_shape in gradient_shapes]

		out_channels = next(
			layer.out_channels for layer in reversed(gradient_components[0]) if hasattr(layer, "out_channels")
		)

		# Add the fcn components to the convolutional components
		for gradient_component in gradient_components:
			gradient_component.append(nn.Flatten(start_dim=-3))
			gradient_component.extend(self._get_fcn_layers(out_channels))

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

			out_channels = next(
				layer.out_channels for layer in reversed(metric_components[0]) if hasattr(layer, "out_channels")
			)

			# Add the fcn components to the convolutional components
			for metric_component in metric_components:
				metric_component.append(nn.Flatten(start_dim=-3))
				metric_component.extend(self._get_fcn_layers(out_channels))

			# Make the metric component
			metric_components = [
				utils.get_net_class_from_layers(metric_component) for metric_component in metric_components
			]
			self.metric_components[key] = [component().to(training.DEVICE) for component in metric_components]

	def _initialize_encoder_component(self, activation_shapes, gradient_shapes, metrics_shapes):
		# Keep track of the size that the encoder needs to take
		encoder_size = 0

		# Helper variable and helper function
		round_multiplier = self.config.simulation.global_rounds + 1
		get_output_size = lambda component: next(
			layer.out_features for layer in reversed(component.layers) if hasattr(layer, "out_features")
		)

		# Size of the label component
		encoder_size += get_output_size(self.label_component)

		# Size of the loss component
		encoder_size += get_output_size(self.loss_component) * round_multiplier

		# Size of the activation components
		activation_output_size = get_output_size(self.activation_components[0])
		encoder_size += len(activation_shapes) * activation_output_size * round_multiplier

		# Size of the gradient components
		gradient_output_size = get_output_size(self.gradient_components[0])
		encoder_size += (sum(gradient_shape[-4] for gradient_shape in gradient_shapes)
						 * gradient_output_size * round_multiplier)

		# Size of the metric components
		if metrics_shapes != {}:
			metric_output_size = get_output_size(next(iter(self.metric_components.values()))[0])
			encoder_size += sum(
				sum(metric_shapes[-4] for metric_shapes in metrics_shapes[key]) for key in metrics_shapes.keys()
			) * metric_output_size * round_multiplier

		layers = []
		in_features_set = False
		raw_encoder_copy = copy.deepcopy(self.config.attack.attack_simulation.model_architecture.encoder)
		for layer in raw_encoder_copy:
			layer_type = getattr(nn, layer["type"])
			parameters = {key: value for key, value in list(layer.items())[1:]}

			# The first linear layer of the attack has the size of the concatenated output of all components
			if not in_features_set and layer_type is nn.Linear:
				in_features_set = True
				parameters = {**parameters, "in_features": encoder_size}

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
			if not dynamic_parameters_set and layer_type is nn.Conv2d:
				parameters["in_channels"] = input_shape[-3]
				parameters["kernel_size"] = input_shape[-2:]
				dynamic_parameters_set = True

			layer = layer_type(**parameters)
			layers.append(layer)
		return layers
