import copy
import itertools
import operator
from functools import reduce
from typing import List, Type

import numpy.typing as npt
import torch
import torch.nn as nn

from src import training
from src.utils import compute_convolution_output_size, get_model_layer_shapes, get_net_class_from_layers
from src.utils.configs import Config


class TransformUnsqueeze(nn.Module):
	def __init__(self):
		super(TransformUnsqueeze, self).__init__()

	def forward(self, x):
		return x.unsqueeze(0)


class AttackModel:
	def __init__(self, fcn, encoder, cnn):
		self.raw_fcn = fcn
		self.raw_encoder = encoder
		self.raw_cnn = cnn

		self.activation_components = None
		self.label_component = None
		self.loss_component = None
		self.gradient_components = None
		self.encoder_component = None

	@staticmethod
	def from_dict(config: dict):
		return AttackModel(**config)

	def get_attack_model(self, run_config: Config):
		# Ensure all the components have been set
		if self.activation_components is None:
			self.activation_components = self.get_activation_components(run_config)
		if self.label_component is None:
			self.label_component = self.get_label_component(run_config)
		if self.loss_component is None:
			self.loss_component = self.get_loss_component(run_config)
		if self.gradient_components is None:
			self.gradient_components = self.get_gradient_components(run_config)
		if self.encoder_component is None:
			self.encoder_component = self.get_encoder_component()

		activation_components = self.activation_components
		label_component = self.label_component
		loss_component = self.loss_component
		gradient_components = self.gradient_components
		encoder_component = self.encoder_component

		class Net(nn.Module):
			def __init__(self) -> None:
				super(Net, self).__init__()
				self.activation_components = [component() for component in activation_components]
				self.label_component = label_component()
				self.loss_component = loss_component()
				self.gradient_components = [component() for component in gradient_components]
				self.encoder_component = encoder_component()

			def forward(self, parameters: List[npt.NDArray], x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
				# The label value for the attack model
				label_value = self.label_component(y.float())

				# Get the prediction and loss of the target model
				model = run_config.get_model()
				if parameters is not None:
					training.set_weights(model, parameters)
				criterion = run_config.get_criterion()
				optimizer = run_config.get_optimizer(model.parameters())
				prediction = model(x)

				# Before finding the gradient get the value of each layer function
				layer_values = []
				for layer in model.layers:
					x = layer(x)
					layer_values.append(x)

				# The activation values for the attack model
				activation_values = [activation_component(layer_value) for activation_component, layer_value in
									 zip(self.activation_components, layer_values)]

				# Get the gradients of the input
				loss = criterion(prediction, y[0])

				# The loss value for the attack model
				loss_value = self.loss_component(loss.unsqueeze(0))

				loss.backward()
				optimizer.step()
				gradients = list(
					itertools.chain.from_iterable(
						[param.grad for param in param_group["params"]] for param_group in optimizer.param_groups
					)
				)

				# The gradient values for the attack model
				gradient_values = []
				for gradient_component, gradient in zip(self.gradient_components, gradients):
					if gradient.ndim == 1:
						gradient = gradient.unsqueeze(0)
					gradient_values.append(gradient_component(gradient))

				# Append the outputs and put in the encoder
				encoder_input_values = torch.cat([*activation_values, label_value, loss_value, *gradient_values], dim=0)
				result = self.encoder_component(encoder_input_values)
				return result

		return Net

	def get_fcn_layers(self, input_size: int) -> List[Type[nn.Module]]:
		"""
		Constructs a list of layers from the raw fcn configuration
		:param input_size: the input size of the fcn
		"""
		layers = []
		in_features_set = False
		raw_fcn_copy = copy.deepcopy(self.raw_fcn)
		for layer in raw_fcn_copy:
			layer_type = getattr(nn, layer["type"])
			parameters = {key: value for key, value in list(layer.items())[1:]}

			# The first linear layer of the attack has as many inputs as there are output classes
			if not in_features_set and layer_type is nn.Linear:
				in_features_set = True
				parameters = {**parameters, "in_features": input_size}

			layer = layer_type(**parameters)
			layers.append(layer)
		return layers

	def get_convolution_layers(self, layer_input_size: int) -> List[Type[nn.Module]]:
		"""
		Constructs a list of layers from the raw cnn configuration
		:param layer_input_size: the input size of the layer which gradient is put in to the convolution layer
		"""
		layers = []
		raw_cnn_copy = copy.deepcopy(self.raw_cnn)
		for layer in raw_cnn_copy:
			layer_type = getattr(nn, layer["type"])
			parameters = {key: value for key, value in list(layer.items())[1:]}
			if layer_type is nn.Conv2d:
				# The kernel of the convolution layer is as wide as the amount of classes
				parameters["kernel_size"].append(layer_input_size)
			layer = layer_type(**parameters)
			layers.append(layer)
		return layers

	def get_activation_components(self, run_config: Config) -> List[Type[nn.Module]]:
		"""
		Constructs a list of activation components from the model. These activation components are used to input
		the outputs of all activation functions in the attacker model via the encoder. The input of each component
		is the output of the activation function, and it will output depending on the configuration of the experiment.
		:param run_config: the configuration of the experiment
		"""
		model = run_config.get_model()

		activation_components = []
		previous_layer_size = None
		for layer in model.layers:
			# The output size of the layer is either indicated by the layer itself,
			# or it is the input size of the previous layer
			if hasattr(layer, "out_features"):
				previous_layer_size = layer.out_features
			elif previous_layer_size is None:
				previous_layer_size = run_config.get_class_count()

			activation_component_layers = self.get_fcn_layers(previous_layer_size)
			activation_component = get_net_class_from_layers(activation_component_layers)
			activation_components.append(activation_component)
		return activation_components

	def get_label_component(self, run_config: Config) -> Type[nn.Module]:
		"""
		Constructs the label component from the model. The label component is used to input
		the expected output of the target model into the attacker model via the encoder. So, only the expected output
		will be put in, and the output is dependent on the configuration of the experiment.
		:param run_config: the configuration of the experiment
		"""
		layers = self.get_fcn_layers(1)
		label_component = get_net_class_from_layers(layers)
		return label_component

	def get_loss_component(self, run_config: Config) -> Type[nn.Module]:
		"""
		Constructs the loss component from the model. The loss component is used to input
		the loss of the target model (w.r.t. the expected output and the prediction) into the attacker model via the
		encoder. So, only the expected output will be put in, and the output is dependent on the configuration
		of the experiment.
		:param run_config: the configuration of the experiment
		"""
		layers = self.get_fcn_layers(1)
		loss_component = get_net_class_from_layers(layers)
		return loss_component

	def get_gradient_components(self, run_config: Config) -> List[Type[nn.Module]]:
		"""
		Constructs the gradient components from the model. The gradient component is used to input the gradients of each
		layer into the attacker model via the encoder. Before going into a fcn, a convolutional component is used to
		get the information of the grid-like data. When the gradient comes from a fcn, the width of the kernel is set
		to the input size of the layer so that the output of the convolutional component is a vector. That vector is
		then put into the fcn component. The output of the fcn component is used as input for the attack encoder.
		:param run_config: the configuration of the experiment
		"""
		# Get the size of each layer, so we know what shape to make the convolutional components
		model = run_config.get_model()
		layer_shapes = get_model_layer_shapes(model, run_config)

		# Construct the gradient components
		gradient_components = []
		for input_shape, output_shape in layer_shapes:
			gradient_component = []

			# The gradients need to be transposed and unsqueezed in our implementation
			gradient_component.append(TransformUnsqueeze())

			# TODO: Fix convolutional layer
			convolutional_layers = self.get_convolution_layers(input_shape[0])
			gradient_component.extend(convolutional_layers)

			# The gradients are flattened after the convolutional component
			gradient_component.append(nn.Flatten(start_dim=0))

			# Construct the fully connected layer component
			# Get the output size of the convolutional component
			out_channels, kernel_size, stride, padding = None, None, None, None
			for convolutional_layer in reversed(convolutional_layers):
				if not hasattr(convolutional_layer, "out_channels"):
					continue
				out_channels = convolutional_layer.out_channels
				kernel_size = convolutional_layer.kernel_size
				stride = convolutional_layer.stride
				padding = convolutional_layer.padding
				break

			gradient_shape = (*output_shape, *input_shape)
			if len(gradient_shape) == 1:
				gradient_shape = (1, *gradient_shape)

			convolution_output_size = compute_convolution_output_size(
				gradient_shape,
				out_channels,
				kernel_size,
				stride,
				padding
			)
			fcn_input_size = reduce(operator.mul, convolution_output_size, 1)
			fcn_layers = self.get_fcn_layers(fcn_input_size)
			gradient_component.extend(fcn_layers)
			gradient_component = get_net_class_from_layers(gradient_component)
			gradient_components.append(gradient_component)
		return gradient_components

	def get_encoder_component(self) -> Type[nn.Module]:
		"""
		Constructs the encoder component for the attacker model. It requires all other components to be set. The encoder
		component is used to combine the outputs of all other components into a single probability that predicts
		if the input is a member.
		"""
		# Get the output size of each component
		output_sizes = []
		components = [*self.activation_components, self.label_component, self.loss_component, *self.gradient_components]
		for component in components:
			net = component()
			for layer in reversed(net.layers):
				if hasattr(layer, "out_features"):
					output_sizes.append(layer.out_features)
					break

		input_size = sum(output_sizes)
		layers = []
		in_features_set = False
		raw_encoder_copy = copy.deepcopy(self.raw_encoder)
		for layer in raw_encoder_copy:
			layer_type = getattr(nn, layer["type"])
			parameters = {key: value for key, value in list(layer.items())[1:]}

			# The first linear layer of the attack has as many inputs as there are output classes
			if not in_features_set and layer_type is nn.Linear:
				in_features_set = True
				parameters = {**parameters, "in_features": input_size}

			layer = layer_type(**parameters)
			layers.append(layer)
		encoder_component = get_net_class_from_layers(layers)
		return encoder_component
