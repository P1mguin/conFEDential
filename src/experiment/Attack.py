import copy
import itertools
import math
import random
from typing import List

import numpy.typing as npt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import src.training as training
from src.experiment import AttackSimulation


class Attack:
	def __init__(
			self,
			data_access: float,
			message_access: str,
			repetitions: int,
			attack_simulation=None
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
	def data_access(self) -> float:
		return self._data_access

	@property
	def repetitions(self) -> int:
		return self._repetitions

	@property
	def attack_simulation(self):
		return self._attack_simulation

	def reset_variables(self):
		self._client_id = None
		self._is_target_member = bool(random.getrandbits(1))
		self._target = None
		self._target_client = None

	def get_message_access_indices(self, client_count) -> List[int]:
		if self._message_access == "client":
			return [self._client_id]
		elif self._message_access == "server":
			return list(range(client_count))
		else:
			return []

	def get_target_sample(self, simulation):
		"""
		Function that selects the target to attack based on the configuration. Samples one random target from all the
		data, returns the target, whether it was taken from the training data (if it is a member), and which client
		index the sample was taken from. In case the target is not a member, the client index will be None.
		"""
		if self._target is not None:
			return self._target, self._is_target_member, self._target_client

		# Generate whether the victim is a member, and if so what client they originate from
		if self._is_target_member:
			train_loaders = simulation.train_loaders
			self._target_client = random.randint(0, len(train_loaders) - 1)
			self._target = random.choice(train_loaders[self._target_client].dataset)
		else:
			# Pick any value from the test dataset
			test_loader = simulation.test_loader
			self._target_client = None
			self._target = random.choice(test_loader.dataset)

		return self._target, self._is_target_member, self._target_client

	def get_membership_inference_attack_dataset(
			self,
			models,
			metrics,
			intercepted_data,
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
		# Helper variables for clarity
		dataset_size = len(intercepted_data)

		# Extract the value that will be used in the attack dataset into separate variables
		features = torch.stack([torch.tensor(value[0][0]) for value in intercepted_data])
		labels = torch.stack([torch.tensor(value[0][1]) for value in intercepted_data])
		is_member = torch.stack([torch.tensor(value[1]) for value in intercepted_data])
		member_origins = torch.tensor([value[2] if value[2] else -1 for value in intercepted_data])

		# Translate the boolean to an integer
		is_member = is_member.int()

		# Translate the labels to be one-hot-encoded
		labels = nn.functional.one_hot(labels)

		# Reshape the metrics so they are 5D and can fit in a gradient component
		metrics = {
			key: [reshape_to_4d(torch.tensor(layer), True).float() for layer in value] for key, value in metrics.items()
		}

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
				batch_member_origins = member_origins[start:end]

				activation_values, gradients, loss = self._precompute_attack_features(
					batch_features, batch_labels, models, simulation
				)

				for j in range(batch_size):
					label = batch_labels[j].float()
					gradient = [layer[j] for layer in gradients]
					activation_value = [layer[j] for layer in activation_values]
					loss_value = loss[j].unsqueeze(-1)
					is_value_member = batch_is_member[j]
					value_origin = batch_member_origins[j]
					yield ((gradient, activation_value, metrics, loss_value, label),
						   is_value_member.float(), value_origin)

		attack_dataset = attack_dataset_generator()
		dataset = GeneratorDataset(attack_dataset, dataset_size)
		attack_dataloader = DataLoader(dataset, batch_size=self._attack_simulation.batch_size, shuffle=True)
		return attack_dataloader

	def _precompute_attack_features(self, features, labels, model_iterations, simulation):
		"""
		Gets the loss, activation functions and gradients for a list of parameters
		"""
		# Get the models from the parameter iterations
		batch_size = len(features)
		models = self._get_models(model_iterations, batch_size, simulation)
		activation_values = self._get_activation_values(models, features, simulation)
		predictions = activation_values[-1]
		losses = self._get_losses(predictions, labels, simulation)
		gradient_values = self._get_gradient_values(losses, models, simulation)
		return activation_values, gradient_values, losses

	def _get_models(self, model_iterations: List[npt.NDArray], batch_size: int, simulation) -> List[List[nn.Module]]:
		"""
		Creates a list of models from a list of parameters from the model iterations
		"""
		global_rounds = simulation.global_rounds

		# Get the aggregated models and the initial model from the iterations
		models = []
		for i in range(global_rounds + 1):
			model = copy.deepcopy(simulation.model)
			model_parameters = [layer[i] for layer in model_iterations]
			training.set_weights(model, model_parameters)
			models.append(model.to(training.DEVICE))
		models = [[copy.deepcopy(model) for model in models] for _ in range(batch_size)]
		return models

	def _get_activation_values(self, models, features, simulation):
		"""
		Gets the activation values from a list of model iterations and features
		"""
		# Expand the features once, so it accounts for several model iterations
		global_rounds = simulation.global_rounds
		features = features.unsqueeze(1).repeat_interleave(global_rounds + 1, dim=1)

		trainable_indices = simulation.model_config.get_trainable_layer_indices()
		layer_count = len(models[0][0].layers)

		def get_activation_values():
			values = features
			for i in range(layer_count):
				values = torch.stack([
					torch.stack([model.layers[i](value) for model, value in zip(model_iterations, value_iterations)])
					for model_iterations, value_iterations in zip(models, values)
				])
				if i not in trainable_indices:
					yield values

		activation_values = list(get_activation_values())
		return activation_values

	def _get_losses(self, predictions, label, simulation):
		"""
		Gets the losses of a list of predictions and a list of labels
		"""
		global_rounds = simulation.global_rounds
		criterion = simulation.criterion
		criterion.reduction = "none"
		loss = torch.stack([criterion(predictions[:, i, :], label.float()) for i in range(global_rounds + 1)])
		loss = loss.view(-1, global_rounds + 1)
		return loss

	def _get_gradient_values(self, losses, models, simulation):
		"""
		Gets the gradients from a list of model iterations and features
		"""
		losses.sum().backward()

		template_model = simulation.model

		def get_gradients():
			for i in range(len(list(template_model.parameters()))):
				layer_gradients = torch.stack([
					torch.stack([
						reshape_to_4d(next(itertools.islice(model.parameters(), i, None)).grad)
						for model in model_iterations
					]) for model_iterations in models
				])
				yield layer_gradients

		gradients = list(get_gradients())
		return gradients


def reshape_to_4d(input_tensor: torch.Tensor, batched: bool = False) -> torch.Tensor:
	if batched:
		unsqueeze_dim = 2
		target_dim = 5
	else:
		unsqueeze_dim = 1
		target_dim = 4

	while input_tensor.ndim > target_dim:
		input_tensor = input_tensor.view(*input_tensor.shape[:(target_dim-1)], -1)
	while input_tensor.ndim < target_dim:
		input_tensor = input_tensor.unsqueeze(unsqueeze_dim)
	return input_tensor

class GeneratorDataset(Dataset):
	def __init__(self, generator, length):
		self._generator = generator
		self._length = length

	def __len__(self):
		return self._length

	def __getitem__(self, index):
		return next(self._generator)
