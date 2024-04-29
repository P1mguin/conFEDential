import copy
from typing import List, Tuple, Type

import torch
import torch.nn as nn

from src import training
from src.utils.configs import Config


class AttackNet(nn.Module):
	def __init__(
			self,
			activation_components: List[Type[nn.Module] | None],
			gradient_components: List[Tuple[Type[nn.Module], Type[nn.Module]] | None],
			loss_component: Type[nn.Module],
			label_component: Type[nn.Module],
			encoder_component: Type[nn.Module],
			run_config: Config
	) -> None:
		super(AttackNet, self).__init__()
		self.activation_components = [
			component().to(training.DEVICE) if component is not None else None for component in activation_components
		]
		self.gradient_components = [
			[component().to(training.DEVICE) for component in components] if components is not None else None
			for components in gradient_components
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
		template_model = self.run_config.get_model()
		batch_size = parameters[0].shape[0]
		models = []
		for i in range(batch_size):
			model = copy.deepcopy(template_model)
			new_state_dict = {key: parameter[i] for key, parameter in zip(model.state_dict().keys(), parameters)}
			model.load_state_dict(new_state_dict)
			# Get the models to where the computer assumes it is
			models.append(model.to(training.DEVICE))
		return models

	def get_activation_values(self, models: List[nn.Module], x: torch.Tensor) -> torch.Tensor:
		"""
		Passes the input through the models and gets the activation values
		:param models: A list of models to pass the input through
		:param x: The input, may be batched in that case the batch size must equal length of models
		"""
		activation_values = []
		for i, model in enumerate(models):
			value = x[i]
			activation_value = []
			# Pass each value through the layer
			for j, layer in enumerate(model.layers):
				value = layer(value)

				if self.activation_components[j] is None:
					# No need to append nothing as these values are to be concatenated into the encoder
					continue

				component_value = self.activation_components[j](value)
				activation_value.append(component_value)
			activation_value = torch.cat(activation_value, dim=0)
			activation_values.append(activation_value)
		activation_values = torch.stack(activation_values)
		return activation_values

	def get_loss_value(
			self,
			models: List[nn.Module],
			x: torch.Tensor,
			y: torch.Tensor
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Calculates the loss and the value of the loss component
		:param models: list of models to compute the loss for
		:param x: the input to compute the loss for, may be batched in that case the batch size must equal length of models
		:param y: the expected output(s)
		"""
		criterion = self.run_config.get_criterion()
		criterion.reduction = "none"
		predictions = torch.stack([model(value) for model, value in zip(models, x)])
		loss = criterion(predictions, y)
		loss_value = self.loss_component(loss.unsqueeze(1))
		return loss, loss_value

	def get_gradients_values(self, losses: torch.Tensor, models: List[nn.Module]) -> torch.Tensor:
		"""
		Calculates the gradient values for the models
		:param losses: the losses of the models for the value that was put in
		:param models: the models for which to compute the gradient component values
		"""
		# Do a backwards step for each model
		optimizers = [self.run_config.get_optimizer(model.parameters()) for model in models]
		# Retain graph is needed to calculate the gradients for all models
		[loss.backward(retain_graph=True) for loss in losses]

		# Get the gradients of all the model for all the layers
		params = [param_group["params"] for optimizer in optimizers for param_group in optimizer.param_groups]
		gradients = [[param.grad for param in model] for model in params]

		# Pair the tuples and align None values with gradient components
		gradients = [list(zip(gradient[::2], gradient[1::2])) for gradient in gradients]
		gradients = [[gradients[i].pop(0) if component is not None else None for component in self.gradient_components]
					 for i in range(len(gradients))]

		# Get the gradient values for each model and layer
		gradient_values = torch.stack([
			torch.cat(
				[
					torch.cat(
						[gradient_component[0](layer[0]), gradient_component[1](layer[1])]
					) for layer, gradient_component in zip(gradient, self.gradient_components) if layer is not None
				]
			) for gradient in gradients
		])
		return gradient_values

	def forward(self, parameters: List[torch.Tensor], x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
		is_batched = x.ndim == 4 or x.ndim == 2
		if not is_batched:
			x = x.unsqueeze(0)
			parameters = [parameter.unsqueeze(0) for parameter in parameters]

		models = self.get_models(parameters)

		label_value = self.get_label_value(y)
		activation_values = self.get_activation_values(models, x)
		loss, loss_value = self.get_loss_value(models, x, y)
		gradient_values = self.get_gradients_values(loss, models)

		encoder_input_values = torch.cat([activation_values, loss_value, label_value, gradient_values], dim=1)
		result = self.encoder_component(encoder_input_values)
		result = result.view(-1)
		if not is_batched:
			result = result.squeeze(0)
		return result
