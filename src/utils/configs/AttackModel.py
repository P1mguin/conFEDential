import copy
import itertools
import operator
from functools import reduce
from typing import List, Set, Tuple, Type

import numpy.typing as npt
import torch
import torch.nn as nn
from torch.nn import Module

from src import training
from src.utils import (get_gradient_shapes, get_layer_shapes, get_net_class_from_layers, get_trainable_layers_indices)
from src.utils.configs import Config


class Expand(nn.Module):
	def __init__(self, times):
		super(Expand, self).__init__()
		self.times = times

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		for _ in range(self.times):
			x = x.unsqueeze(0)
		return x


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
		self.initialize_components(run_config)

		activation_components = self.activation_components
		label_component = self.label_component
		loss_component = self.loss_component
		gradient_components = self.gradient_components
		encoder_component = self.encoder_component

		# Merge the activation components and gradient components into a single layer
		layer_components = [
			(activation,) if gradient is None else gradient
			for activation, gradient in zip(activation_components, gradient_components)
		]
		trainable_layers_indices = get_trainable_layers_indices(run_config.get_model())

		class Net(nn.Module):
			def __init__(self) -> None:
				super(Net, self).__init__()
				self.layer_components = [[component() for component in components] for components in layer_components]
				self.label_component = label_component()
				self.loss_component = loss_component()
				self.encoder_component = encoder_component()

			def forward(self, parameters: List[npt.NDArray], x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
				# In case the tensor is unbatched, f.e. 3dim, expand
				is_unbatched = x.ndim == 3 or x.ndim == 1
				if is_unbatched:
					x = x.unsqueeze(0)

				# The label value for the attack model
				label_value = self.label_component(y.float())

				# Get the prediction and loss of the target model
				model = run_config.get_model()
				if parameters is not None:
					training.set_weights(model, parameters)
				criterion = run_config.get_criterion()
				optimizer = run_config.get_optimizer(model.parameters())
				prediction = model(x)

				# Before finding the gradient get the value of each activation function
				activation_values = []
				for i, layer in enumerate(model.layers):
					x = layer(x)
					if i in trainable_layers_indices:
						continue
					if is_unbatched:
						activation_values.append(x.squeeze(0))
					else:
						activation_values.append(x)

				# The loss value for the attack model, unsqueeze to make it a tensor
				loss = criterion(prediction, y)
				loss_value = self.loss_component(loss.unsqueeze(0))

				# Get the gradients of the input
				loss.backward()
				optimizer.step()
				gradients = list(
					itertools.chain.from_iterable(
						[param.grad for param in param_group["params"]] for param_group in optimizer.param_groups
					)
				)

				# Combine the gradients to be a list of tuples (weights, biases)
				gradients = list(zip(gradients[::2], gradients[1::2]))

				# Get the layer_component values
				layer_values = []
				for layer_component in self.layer_components:
					if len(layer_component) == 2:
						weight_component, bias_component = layer_component
						weight, bias = gradients.pop(0)
						layer_values.append(weight_component(weight))
						layer_values.append(bias_component(bias))
					else:
						value = activation_values.pop(0)
						layer_values.append(layer_component[0](value))

				# Append the outputs and put in the encoder
				encoder_input_values = torch.cat([*layer_values, loss_value, label_value], dim=0)
				result = self.encoder_component(encoder_input_values)
				return result

		return Net

	def initialize_components(self, run_config: Config) -> None:
		"""
		Initializes the components of the attack model
		:param run_config: the configuration of the experiment
		"""
		trainable_layers_indices = get_trainable_layers_indices(run_config.get_model())

		if self.activation_components is None:
			self.activation_components = self.get_activation_components(trainable_layers_indices, run_config)
		if self.label_component is None:
			self.label_component = self.get_label_component()
		if self.loss_component is None:
			self.loss_component = self.get_loss_component()
		if self.gradient_components is None:
			self.gradient_components = self.get_gradient_components(trainable_layers_indices, run_config)
		if self.encoder_component is None:
			self.encoder_component = self.get_encoder_component(run_config)

	def get_fcn_layers(self, input_size: int) -> List[Type[nn.Module]]:
		"""
		Constructs a list of layers from the raw fcn configuration
		:param input_size: the input size of the fcn
		"""
		layers = []
		raw_fcn_copy = copy.deepcopy(self.raw_fcn)
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

	def get_convolution_layers(self, input_shape: Tuple[int, ...]) -> List[Type[nn.Module]]:
		"""
		Constructs a list of layers from the raw cnn configuration
		:param input_shape: the input shape of the layer which gradient is put in to the convolution layer
		"""
		layers = []
		raw_cnn_copy = copy.deepcopy(self.raw_cnn)
		dynamic_parameters_set = False
		for layer in raw_cnn_copy:
			# Get the static parameters
			layer_type = getattr(nn, layer["type"])
			parameters = {key: value for key, value in list(layer.items())[1:]}

			# The kernel of the first convolution layer is as wide as the width/breadth of the input shape
			if not dynamic_parameters_set and layer_type is nn.Conv2d:
				# The convolution layer is either for the weights or bias, so length of 4 or 2, or length of 1
				if len(input_shape) == 4:
					parameters["in_channels"] = input_shape[1]
				else:
					parameters["in_channels"] = 1
				parameters["kernel_size"].append(input_shape[-1])
				dynamic_parameters_set = True

			layer = layer_type(**parameters)
			layers.append(layer)
		return layers

	def get_activation_components(self, trainable_layers_indices: Set[int], run_config: Config) \
			-> List[Type[Module] | None]:
		"""
		Constructs a list of activation components from the model. These activation components are used to input
		the outputs of all activation functions in the attacker model via the encoder. The input of each component
		is the output of the activation function, and it will output depending on the configuration of the experiment.
		:param trainable_layers_indices: the indices of the trainable layers of the target model
		:param run_config: the configuration of the experiment
		"""
		model = run_config.get_model()
		layer_output_shapes = get_layer_shapes(model, run_config)
		activation_components = []
		for i in range(len(model.layers)):
			# Trainable layers are accounted for in the gradients components
			if i in trainable_layers_indices:
				activation_components.append(None)
				continue

			layer_output_shape = layer_output_shapes[i + 1]
			is_flattened = len(layer_output_shape) > 1
			if is_flattened:
				layer_output_shape = (reduce(operator.mul, layer_output_shape, 1),)

			activation_component = self.get_fcn_layers(layer_output_shape[0])

			if is_flattened:
				# Prepend a flattening layer to the activation component
				activation_component.insert(0, nn.Flatten(start_dim=0))

			activation_component = get_net_class_from_layers(activation_component)
			activation_components.append(activation_component)
		return activation_components

	def get_label_component(self) -> Type[nn.Module]:
		"""
		Constructs the label component from the model. The label component is used to input
		the expected output of the target model into the attacker model via the encoder. So, only the expected output
		will be put in, and the output is dependent on the configuration of the experiment.
		"""
		layers = self.get_fcn_layers(1)
		label_component = get_net_class_from_layers(layers)
		return label_component

	def get_loss_component(self) -> Type[nn.Module]:
		"""
		Constructs the loss component from the model. The loss component is used to input
		the loss of the target model (w.r.t. the expected output and the prediction) into the attacker model via the
		encoder. So, only the expected output will be put in, and the output is dependent on the configuration
		of the experiment.
		"""
		layers = self.get_fcn_layers(1)
		loss_component = get_net_class_from_layers(layers)
		return loss_component

	def get_gradient_components(self, trainable_layers_indices: Set[int], run_config: Config) -> list[
		tuple[Type[Module], Type[Module]] | None]:
		"""
		Constructs the gradient components from the model. The gradient component is used to input the gradients of each
		layer into the attacker model via the encoder. Before going into a fcn, a convolutional component is used to
		get the information of the grid-like data. When the gradient comes from a fcn, the width of the kernel is set
		to the input size of the layer so that the output of the convolutional component is a vector. That vector is
		then put into the fcn component. The output of the fcn component is used as input for the attack encoder.
		:param trainable_layers_indices: the indices of the trainable layers of the target model
		:param run_config: the configuration of the experiment
		"""
		# Get the size of each layer, so we know what shape to make the convolutional components
		model = run_config.get_model()
		gradient_shapes = get_gradient_shapes(model)
		gradient_components = []
		for i in range(len(model.layers)):
			# Non-trainable layers are accounted for in the activation components
			if i not in trainable_layers_indices:
				gradient_components.append(None)
				continue

			weights_shape, bias_shape = gradient_shapes.pop(0)

			weights_convolution_component = []

			# We unsqueeze the weights of a FCL since it does not have any channels by default
			is_fcl = len(weights_shape) == 2
			if is_fcl:
				weights_convolution_component.append(Expand(1))

			weights_convolution_component.extend(self.get_convolution_layers(weights_shape))

			# The bias is first unsqueezed and then put into the convolutional component
			bias_convolution_component = [Expand(2)]
			bias_convolution_component.extend(self.get_convolution_layers(bias_shape))

			if is_fcl:
				weights_convolution_component.append(nn.Flatten(-2))
			else:
				weights_convolution_component.append(nn.Flatten(-3))
			bias_convolution_component.append(nn.Flatten(-3))

			# Get the output size of both convolutional components
			bias_net = get_net_class_from_layers(bias_convolution_component)()
			out_channels = [param.shape for _, param in bias_net.named_parameters()][0][0]

			if is_fcl:
				weights_size = weights_shape[-2]
			else:
				weights_size = weights_shape[-2] * out_channels
			bias_size = out_channels

			# Create the linear components and append
			weights_convolution_component.extend(self.get_fcn_layers(weights_size))
			bias_convolution_component.extend(self.get_fcn_layers(bias_size))

			# Flatten the output of the fcn components
			weights_convolution_component.append(nn.Flatten(0))
			bias_convolution_component.append(nn.Flatten(0))

			weight_net = get_net_class_from_layers(weights_convolution_component)
			bias_net = get_net_class_from_layers(bias_convolution_component)
			gradient_components.append((weight_net, bias_net))
		return gradient_components

	def get_encoder_component(self, run_config: Config) -> Type[nn.Module]:
		"""
		Constructs the encoder component for the attacker model. It requires all other components to be set. The encoder
		component is used to combine the outputs of all other components into a single probability that predicts
		if the input is a member.
		:param run_config: the configuration of the experiment
		"""
		model = run_config.get_model()
		# First go over the activation and gradient components
		# The gradient components will add the bias and weights in one sweep
		layer_components = [
			activation if gradient is None else gradient[0]
			for activation, gradient in zip(self.activation_components, self.gradient_components)
		]
		trainable_indices = get_trainable_layers_indices(run_config.get_model())

		# Get the output size of each component
		input_size = 0
		for i, component in enumerate(layer_components):
			net = component()

			# An FCL component will start by expanding, we need this distinction to know the output shape of the training
			# Component
			is_training = i in trainable_indices
			is_fcl = is_training and isinstance(net.layers[0], Expand)

			# For non-training layers return the amount of output features, as their input is one dimensional
			out_features = None
			for layer in reversed(net.layers):
				if hasattr(layer, "out_features"):
					out_features = layer.out_features
					break
			if not is_training:
				input_size += out_features
				continue

			# For an FCL the extra channels are the amount of output kernels of the last convolution layer,
			# for a convolutional layer the extra channels are the amount of output channels of the convolution layer
			convolution_out_features = None
			for layer in reversed(net.layers):
				if hasattr(layer, "out_channels"):
					convolution_out_features = layer.out_channels
					break

			if is_fcl:
				input_size += out_features * convolution_out_features
			else:
				input_size += out_features * model.layers[i].out_channels

			# Also add for the bias
			input_size += out_features

		# Add the out features two times for the loss and label component
		input_size += out_features * 2

		layers = []
		in_features_set = False
		raw_encoder_copy = copy.deepcopy(self.raw_encoder)
		for layer in raw_encoder_copy:
			layer_type = getattr(nn, layer["type"])
			parameters = {key: value for key, value in list(layer.items())[1:]}

			# The first linear layer of the attack has the size of the concatenated output of all components
			if not in_features_set and layer_type is nn.Linear:
				in_features_set = True
				parameters = {**parameters, "in_features": input_size}

			layer = layer_type(**parameters)
			layers.append(layer)
		encoder_component = get_net_class_from_layers(layers)
		return encoder_component
