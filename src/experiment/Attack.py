import copy
import itertools
import math
import random
from typing import List, Tuple

import numpy.typing as npt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import src.training as training
import src.utils as utils
from src.experiment import AttackSimulation, Simulation


class Attack:
	def __init__(
			self,
			data_access: str,
			message_access: str,
			repetitions: int,
			attack_simulation = None
	):
		self._data_access = data_access
		self._message_access = message_access
		self._repetitions = repetitions
		self._attack_simulation = attack_simulation

		self._client_id = None
		self._is_target_member = bool(random.getrandbits(1))
		self._target = None
		self._target_client = None

	def __str__(self):
		result = "Attack:"
		result += f"\n\tdata_access: {self._data_access}"
		result += f"\n\tmessage_access: {self._message_access}"
		result += f"\n\trepetitions: {self._repetitions}"
		result += "\n\t{}".format("\n\t".join(str(self._attack_simulation).split("\n")))
		return result

	def __repr__(self):
		result = "Attack("
		result += f"data_access={self._data_access}, "
		result += f"message_access={self._message_access}, "
		result += f"repetitions={self._repetitions}, "
		result += f"{repr(self._attack_simulation)}"
		result += ")"
		return result

	@staticmethod
	def from_dict(config: dict) -> 'Attack':
		attack_simulation = AttackSimulation.from_dict(config['attack_simulation'])
		return Attack(
			data_access=config['data_access'],
			message_access=config['message_access'],
			repetitions=config['repetitions'],
			attack_simulation=attack_simulation,
		)

	@property
	def client_id(self) -> int:
		return self._client_id

	@client_id.setter
	def client_id(self, client_id: int):
		self._client_id = client_id

	@property
	def data_access(self) -> str:
		return self._data_access

	@property
	def repetitions(self) -> int:
		return self._repetitions

	@property
	def attack_simulation(self):
		return self._attack_simulation

	def get_data_access_indices(self, client_count) -> List[int]:
		if self._data_access == "client":
			return [self._client_id]
		elif self._data_access == "all":
			return list(range(client_count))

	def get_target_sample(self, simulation):
		"""
		Function that selects the target to attack based on the configuration. In case the attacker has access to all
		data, it will be any sample from the dataset. If they only have access to one client, the sample will come from
		anything but the client. Returns the federation client id from which the attack target was taken, and the
		attack target
		"""
		if self._target is not None:
			return self._target

		# If the attacker has access to all data, pick any random sample
		# Otherwise pick any random sample from anything other than the client
		train_loaders = simulation.train_loaders
		test_loader = simulation.test_loader

		if self._is_target_member:
			# Select a client to target and then a sample
			possible_targets = list(range(len(train_loaders)))
			if self.data_access == "client":
				possible_targets.remove(self._client_id)
			self._target_client = random.choice(possible_targets)

			self._target = random.choice(train_loaders[self._target_client].dataset)
		else:
			# Pick any value from the test dataset
			self._target_client = None
			self._target = random.choice(test_loader.dataset)

		return self._target_client, self._target

	def get_attack_dataset(
			self,
			server_aggregates: Tuple[List[npt.NDArray], dict],
			attack_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
			simulation
	) -> DataLoader:
		"""
		A function that takes a simulation of a federated learning application, a list of intercepted data:
		(features, labels, and a boolean 'is used in training') and the description of the simulation and creates
		a dataloader that can be used to train the attacker model. Each sample in the data loader is
		the activation function, the gradient and the loss of the sample per model aggregate iteration and the label:
		whether the model was trained on the client
		:return:
		"""
		model_iterations = server_aggregates[0]
		metric_iterations = server_aggregates[1]

		# Reshape each value in the metrics to be 5D
		for key, value in metric_iterations.items():
			for i, layer in enumerate(value):
				layer_value = torch.tensor(layer)
				while layer_value.ndim < 5:
					layer_value = layer_value.unsqueeze(1)

				metric_iterations[key][i] = layer_value.float()

		features = torch.stack([value[0] for value in attack_data])
		labels = torch.stack([value[1] for value in attack_data])
		is_member = torch.stack([value[2] for value in attack_data])

		batch_size = self._attack_simulation.batch_size
		batch_amount = math.ceil(len(features) / batch_size)

		def attack_dataset_generator():
			for i in range(batch_amount):
				start = i * batch_size
				end = (i + 1) * batch_size

				# Stack the information of the shadow_model_simulation in tensors
				batch_features = features[start:end]
				batch_labels = labels[start:end]
				batch_is_member = is_member[start:end]

				activation_values, gradients, loss = self._get_model_information_from_iterations(
					batch_features, batch_labels, model_iterations, simulation
				)

				for j in range(batch_size):
					activation_value = [layer[j] for layer in activation_values]
					gradient = [layer[j] for layer in gradients]
					loss_value = loss[j]
					is_value_member = batch_is_member[j]
					label = batch_labels[j].float()
					yield (activation_value, gradient, loss_value, label, metric_iterations), is_value_member

		attack_dataset = attack_dataset_generator()
		dataset = GeneratorDataset(attack_dataset, len(attack_data))
		attack_dataloader = DataLoader(dataset, batch_size=self._attack_simulation.batch_size)
		return attack_dataloader

	def _get_model_information_from_iterations(self, features, labels, model_iterations, simulation):
		"""
		Gets the loss, activation functions and gradients for a list of parameters
		"""
		features, labels = torch.tensor(features).to(training.DEVICE), torch.tensor(labels).to(training.DEVICE)

		# Get the models from the parameter iterations
		batch_size = len(features)
		models = self._get_models(model_iterations, batch_size, simulation)
		activation_values = self._get_activation_values(models, features)
		predictions = activation_values[-1]
		losses = self._get_losses(predictions, labels, simulation)
		gradient_values = self._get_gradient_values(losses, models)
		return activation_values, gradient_values, losses

	def _get_models(self, model_iterations: List[npt.NDArray], batch_size: int, simulation) -> List[List[nn.Module]]:
		"""
		Creates a list of models from a list of parameters from the model iterations
		"""
		iteration_count = model_iterations[0].shape[0]

		models = []
		for i in range(iteration_count):
			model = copy.deepcopy(simulation.model)
			new_state_dict = {
				key: torch.tensor(parameter[i]) for key, parameter in zip(model.state_dict().keys(), model_iterations)
			}
			model.load_state_dict(new_state_dict)
			models.append(model.to(training.DEVICE))
		models = [[copy.deepcopy(model) for model in models] for _ in range(batch_size)]
		return models

	def _get_activation_values(self, models, features):
		"""
		Gets the activation values from a list of model iterations and features
		"""
		trainable_indices = utils.get_trainable_layers_indices(models[0][0])
		layer_count = len(models[0][0].layers)

		def get_activation_values():
			value = features
			for i in range(layer_count):
				value = torch.stack(
					[
						torch.stack([iteration.layers[i](value[j][k]) for k, iteration in enumerate(iterations)])
						for j, iterations in enumerate(models)
					]
				)
				if i not in trainable_indices:
					yield value

		activation_values = list(get_activation_values())
		return activation_values

	def _get_losses(self, predictions, label, simulation):
		"""
		Gets the losses of a list of predictions and a list of labels
		"""
		iteration_count = predictions.shape[1]
		criterion = simulation.criterion
		criterion.reduction = "none"
		loss = torch.stack([criterion(predictions[i], label[i]) for i in range(predictions.shape[0])])
		return loss

	def _get_gradient_values(self, losses, models):
		"""
		Gets the gradients from a list of model iterations and features
		"""
		losses.sum().backward()
		trainable_layer_count = len(list(models[0][0].parameters()))

		def get_gradients():
			for i in range(trainable_layer_count):
				def reshape_to_4d(input_tensor: torch.Tensor) -> torch.Tensor:
					while input_tensor.ndim < 4:
						input_tensor = input_tensor.unsqueeze(0)
					return input_tensor

				layer_gradients = torch.stack([
					torch.stack(
						[reshape_to_4d(next(itertools.islice(iteration.parameters(), i, None)).grad)
						 for iteration in iterations]
					) for iterations in models
				])
				yield layer_gradients

		gradients = list(get_gradients())
		return gradients


class GeneratorDataset:
	def __init__(self, generator, length):
		self._generator = generator
		self._length = length

	def __len__(self):
		return self._length

	def __getitem__(self, index):
		return next(self._generator)
