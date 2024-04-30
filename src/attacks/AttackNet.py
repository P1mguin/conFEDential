import itertools
from typing import List, Tuple, Type

import torch
import torch.nn as nn

from src import training
from src.utils import get_trainable_layers_indices
from src.utils.configs import AttackConfig


class AttackNet(nn.Module):
	def __init__(
			self,
			activation_components: List[Type[nn.Module]],
			gradient_components: List[Tuple[Type[nn.Module], Type[nn.Module]]],
			loss_component: Type[nn.Module],
			label_component: Type[nn.Module],
			encoder_component: Type[nn.Module],
			run_config: AttackConfig
	) -> None:
		super(AttackNet, self).__init__()
		self.activation_components = [component().to(training.DEVICE) for component in activation_components]
		self.gradient_components = [
			component().to(training.DEVICE) for components in gradient_components for component in components
		]
		self.label_component = label_component().to(training.DEVICE)
		self.loss_component = loss_component().to(training.DEVICE)
		self.encoder_component = encoder_component().to(training.DEVICE)
		self.run_config = run_config

	def get_label_value(self, y: torch.Tensor) -> torch.Tensor:
		y = y.unsqueeze(1).float()
		return self.label_component(y)

	def get_models(self, parameters: List[torch.Tensor]) -> List[nn.Module]:
		"""
		Creates a list of models from a list of parameters
		:param parameters: the parameters of the model, where each index in the parameter are the parameters of the
		model layer. The element at that index may be batched
		"""
		# Create all the models
		batch_size = parameters[0].shape[0]
		models = []
		for i in range(batch_size):
			model = self.run_config.get_model()
			new_state_dict = {key: parameter[i] for key, parameter in zip(model.state_dict().keys(), parameters)}
			model.load_state_dict(new_state_dict)
			# Get the models to where the computer assumes it is
			models.append(model.to(training.DEVICE))

		return models

	def get_activation_values(self, models: List[nn.Module], x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Passes the input through the models and gets the activation values
		:param models: A list of models to pass the input through
		:param x: The input, may be batched in that case the batch size must equal length of models
		"""
		model = self.run_config.get_model()
		trainable_indices = get_trainable_layers_indices(model)
		layer_count = len(model.layers)

		def get_activation_values():
			value = x
			for i in range(layer_count):
				value = torch.stack([model.layers[i](value[j]) for j, model in enumerate(models)])
				if i not in trainable_indices:
					yield value

		activation_values = list(get_activation_values())

		activation_component_values = torch.cat(
			[component(activation_value) for activation_value, component in
			 zip(activation_values, self.activation_components)],
			dim=1
		)
		return activation_component_values, activation_values[-1]

	def get_loss_value(
			self,
			predictions: torch.Tensor,
			y: torch.Tensor
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Calculates the loss and the value of the loss component
		:param predictions: the predictions to compute the loss for size must be equal to that of y
		:param y: the expected output(s)
		"""
		criterion = self.run_config.get_criterion()
		criterion.reduction = "none"
		loss = criterion(predictions, y)
		loss_value = self.loss_component(loss.unsqueeze(1))
		return loss, loss_value

	def get_gradients_values(self, losses: torch.Tensor, models: List[nn.Module]) -> torch.Tensor:
		"""
		Calculates the gradient values for the models
		:param losses: the losses of the models for the value that was put in
		:param models: the models for which to compute the gradient component values
		"""
		# Get the gradients of all the model for all the layers
		losses.sum().backward()
		trainable_layer_count = len(list(models[0].parameters()))

		def get_gradients():
			for i in range(trainable_layer_count):
				def reshape_to_4d(input_tensor: torch.Tensor) -> torch.Tensor:
					while input_tensor.ndim < 4:
						input_tensor = input_tensor.unsqueeze(0)
					return input_tensor

				layer_gradients = torch.stack(
					list(reshape_to_4d(next(itertools.islice(model.parameters(), i, None)).grad) for model in models))
				yield layer_gradients

		gradients = list(get_gradients())

		# Get the gradient values for each model and layer
		gradient_values = torch.cat(
			[component(gradient) for (gradient, component) in zip(gradients, self.gradient_components)],
			dim=1
		)
		return gradient_values

	def forward(self, parameters: List[torch.Tensor], x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		is_batched = x.ndim == 4 or x.ndim == 2
		if not is_batched:
			x = x.unsqueeze(0)
			parameters = [parameter.unsqueeze(0) for parameter in parameters]

		models = self.get_models(parameters)
		label_value = self.get_label_value(y)
		activation_values, predictions = self.get_activation_values(models, x)
		loss, loss_value = self.get_loss_value(predictions, y)
		gradient_values = self.get_gradients_values(loss, models)
		encoder_input_values = torch.cat([activation_values, loss_value, label_value, gradient_values], dim=1)
		result = self.encoder_component(encoder_input_values)
		result = result.view(-1)
		if not is_batched:
			result = result.squeeze(0)

		return result
