from typing import Dict, List, Sequence

import torch
import torch.nn as nn

import src.attacks.component_constructor as component_constructor
from src import training


class AttackNet(nn.Module):
	def __init__(
			self,
			config,
			gradient_shapes: List[Sequence[int]],
			activation_shapes: List[Sequence[int]],
			metrics_shapes: Dict[str, List[Sequence[int]]],
			label_shape: Sequence[int],
			output_size: int
	):
		super(AttackNet, self).__init__()
		self.config = config

		model_config = config.attack.attack_simulation.model_architecture
		if model_config.use_label:
			self.label_component = None
		if model_config.use_loss:
			self.loss_component = None
		if model_config.use_activation:
			self.activation_components = None
		if model_config.use_gradient:
			self.gradient_components = None
		if model_config.use_metrics:
			self.metric_components = None
		self.encoder_component = None

		self._initialize_components(gradient_shapes, activation_shapes, metrics_shapes, label_shape, output_size)

	def forward(self, gradients, activation_values, metrics, loss, label):
		batch_size = loss.size(0)

		model_config = self.config.attack.attack_simulation.model_architecture
		label = self.label_component(label) if model_config.use_label else torch.empty(0, device=training.DEVICE)
		loss = self.loss_component(loss) if model_config.use_loss else torch.empty(0, device=training.DEVICE)
		activation = torch.cat([
			activation_component(activation_value).view(batch_size, -1)
			for activation_component, activation_value in zip(self.activation_components, activation_values)
		], dim=1) if model_config.use_activation else torch.empty(0, device=training.DEVICE)

		# The input shape of the gradients and metrics is:
		# (in_channels/in_features, out_channels/1, kernel_width/1, kernel_height/out_features)
		if model_config.use_gradient:
			gradients = [gradient.view(-1, *gradient.shape[-3:]) for gradient in gradients]
			gradients = torch.cat([
				gradient_component(gradient).view(batch_size, -1)
				for gradient_component, gradient in zip(self.gradient_components, gradients)
			], dim=1)
		else:
			gradients = torch.empty(0, device=training.DEVICE)

		if model_config.use_metrics:
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
		else:
			metrics = [torch.empty(0, device=training.DEVICE) for _ in metrics.keys()]

		# Concatenate the activation, label, loss, gradient and metric components
		encoder_input = torch.cat((label, loss, activation, gradients, *metrics), dim=1)
		prediction = self.encoder_component(encoder_input).squeeze(1)
		return prediction

	def _initialize_components(
			self,
			gradient_shapes: List[Sequence[int]],
			activation_shapes: List[Sequence[int]],
			metrics_shapes: Dict[str, List[Sequence[int]]],
			label_shape: Sequence[int],
			output_size: int
	) -> None:
		model_config = self.config.attack.attack_simulation.model_architecture
		if model_config.use_label:
			self.label_component = component_constructor.construct_label_component(self.config, label_shape)
		if model_config.use_loss:
			self.loss_component = component_constructor.construct_loss_component(self.config)
		if model_config.use_activation:
			self.activation_components = component_constructor.construct_activation_component(
				self.config,
				activation_shapes
			)
		if model_config.use_gradient:
			self.gradient_components = component_constructor.construct_gradient_component(self.config, gradient_shapes)
		if model_config.use_metrics:
			self.metric_components = component_constructor.construct_metric_component(self.config, metrics_shapes)
		self._initialize_encoder_component(activation_shapes, gradient_shapes, metrics_shapes, output_size)

	def _initialize_encoder_component(
			self,
			activation_shapes: List[Sequence[int]],
			gradient_shapes: List[Sequence[int]],
			metrics_shapes: Dict[str, List[Sequence[int]]],
			output_size: int,
	) -> None:
		encoder_size = self._get_encoder_input_size(activation_shapes, gradient_shapes, metrics_shapes)

		self.encoder_component = component_constructor.construct_encoder_component(
			self.config,
			encoder_size,
			output_size
		)

	def _get_encoder_input_size(
			self,
			activation_shapes: List[Sequence[int]],
			gradient_shapes: List[Sequence[int]],
			metrics_shapes: Dict[str, List[Sequence[int]]],
	):
		# Keep track of the size that the encoder needs to take
		encoder_size = 0

		model_config = self.config.attack.attack_simulation.model_architecture

		# Helper variable and helper function
		round_multiplier = activation_shapes[0][0]
		get_output_size = lambda component: next(
			layer.out_features for layer in reversed(component) if hasattr(layer, "out_features")
		)

		# Size of the label component
		if model_config.use_label:
			encoder_size += get_output_size(self.label_component)

		# Size of the loss component
		if model_config.use_loss:
			encoder_size += get_output_size(self.loss_component) * round_multiplier

		# Size of the activation components
		if model_config.use_activation:
			activation_output_size = get_output_size(self.activation_components[0])
			encoder_size += len(activation_shapes) * activation_output_size * round_multiplier

		# Size of the gradient components
		if model_config.use_gradient:
			gradient_output_size = get_output_size(self.gradient_components[0])
			encoder_size += (sum(gradient_shape[-4] for gradient_shape in gradient_shapes)
							 * gradient_output_size * round_multiplier)

		# Size of the metric components
		if model_config.use_metrics and metrics_shapes != {}:
			metric_output_size = get_output_size(next(iter(self.metric_components.values()))[0])
			encoder_size += sum(
				sum(metric_shapes[-4] for metric_shapes in metrics_shapes[key]) for key in metrics_shapes.keys()
			) * metric_output_size * round_multiplier
		return encoder_size


class MembershipNet(AttackNet):
	def __init__(
			self,
			config,
			gradient_shapes: List[Sequence[int]],
			activation_shapes: List[Sequence[int]],
			metrics_shapes: Dict[str, List[Sequence[int]]],
			label_shape: Sequence[int],
	):
		super(MembershipNet, self).__init__(
			config,
			gradient_shapes,
			activation_shapes,
			metrics_shapes,
			label_shape,
			1
		)


class OriginNet(AttackNet):
	def __init__(
			self,
			config,
			gradient_shapes: List[Sequence[int]],
			activation_shapes: List[Sequence[int]],
			metrics_shapes: Dict[str, List[Sequence[int]]],
			label_shape: Sequence[int],
			client_amount: int
	):
		super(OriginNet, self).__init__(
			config,
			gradient_shapes,
			activation_shapes,
			metrics_shapes,
			label_shape,
			client_amount
		)
