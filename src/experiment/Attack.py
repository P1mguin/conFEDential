import math
import random
from logging import INFO
from typing import List

import numpy as np
import torch
import torch.nn as nn
import wandb
from flwr.common import log
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import src.training as training
from src import utils
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

	def membership_inference_attack_model(self, attack_model, attack_loader, test_loader, wandb_kwargs):
		wandb_kwargs = self._get_wandb_kwargs("membership-inference", wandb_kwargs)
		wandb.init(**wandb_kwargs)

		# Split the dataset into training, validation and testing
		train_loader, validation_loader = utils.split_dataloader(attack_loader, 0.85)

		# Set initial parameters and get initial performance
		i = 0
		val_roc_auc, val_fpr, val_tpr = self._test_model(attack_model, validation_loader)
		test_roc_auc, test_fpr, test_tpr = self._test_model(attack_model, test_loader)

		# Log the initial performance
		log(INFO, f"Initial performance: validation auc {val_roc_auc}, test auc {test_roc_auc}")
		self._log_results(
			[val_roc_auc, test_roc_auc],
			[val_fpr, test_fpr],
			[val_tpr, test_tpr],
			["Validation", "Test"],
			step=-1
		)

		# Get training algorithm variables and set model in training mode
		attack_model = attack_model.to(training.DEVICE)
		attack_model.train()
		criterion = nn.BCELoss()
		optimizer = self._attack_simulation.get_optimizer(attack_model.parameters())

		# Training loop with early stopping
		previous_val_roc_auc = 0.
		total_buffer_iterations = 5
		buffer_iterations = total_buffer_iterations
		while val_roc_auc > previous_val_roc_auc or buffer_iterations > 0:
			if val_roc_auc <= previous_val_roc_auc or buffer_iterations < total_buffer_iterations:
				log(INFO, f"Early stopping: {buffer_iterations} iterations left")
				buffer_iterations -= 1

			previous_val_roc_auc = val_roc_auc
			predictions = torch.Tensor()
			is_members = torch.Tensor()
			for (
					gradient,
					activation_value,
					metrics,
					loss_value,
					label
			), is_value_member, _ in tqdm(train_loader, leave=True):
				gradient = [layer.to(training.DEVICE) for layer in gradient]
				activation_value = [layer.to(training.DEVICE) for layer in activation_value]
				metrics = {key: [layer.to(training.DEVICE) for layer in value] for key, value in metrics.items()}
				loss_value = loss_value.to(training.DEVICE)
				label = label.to(training.DEVICE)

				prediction = attack_model(gradient, activation_value, metrics, loss_value, label)
				predictions = torch.cat((predictions, prediction))
				is_members = torch.cat((is_members, is_value_member))

				optimizer.zero_grad()
				loss = criterion(prediction, is_value_member)
				loss.backward()
				optimizer.step()

			# Remove the gradients before testing performance
			predictions = predictions.detach()
			is_members = is_members.detach()

			# Get the performance after the epoch
			train_fpr, train_tpr, _ = roc_curve(is_members, predictions)
			train_roc_auc = auc(train_fpr, train_tpr)
			test_roc_auc, test_fpr, test_tpr = self._test_model(attack_model, test_loader)
			val_roc_auc, val_fpr, val_tpr = self._test_model(attack_model, validation_loader)

			# Log the performance
			log(INFO, f"Epoch {i}: train auc {train_roc_auc}, validation auc {val_roc_auc}, test auc {test_roc_auc}")
			self._log_results(
				[train_roc_auc, val_roc_auc, test_roc_auc],
				[train_fpr, val_fpr, test_fpr],
				[train_tpr, val_tpr, test_tpr],
				["Train", "Validation", "Test"],
				step=i
			)

			i += 1
		wandb.finish()

	def _test_model(self, model, dataloader):
		# Set the model on the right device and put it in testing mode
		model = model.to(training.DEVICE)
		model.eval()

		predictions = torch.Tensor()
		is_members = torch.Tensor()
		with torch.no_grad():
			for (
					gradient,
					activation_value,
					metrics,
					loss_value,
					label
			), is_value_member, _ in tqdm(dataloader, leave=True):
				gradient = [layer.to(training.DEVICE) for layer in gradient]
				activation_value = [layer.to(training.DEVICE) for layer in activation_value]
				metrics = {key: [layer.to(training.DEVICE) for layer in value] for key, value in metrics.items()}
				loss_value = loss_value.to(training.DEVICE)
				label = label.to(training.DEVICE)

				prediction = model(gradient, activation_value, metrics, loss_value, label)

				predictions = torch.cat((predictions, prediction))
				is_members = torch.cat((is_members, is_value_member))

		fpr, tpr, _ = roc_curve(is_members, predictions)
		roc_auc = auc(fpr, tpr)
		return roc_auc, fpr, tpr

	def _log_results(self, roc_aucs, fprs, tprs, titles, step: int):
		wandb_log = {}
		# Add the auc to the wandb log
		for i in range(len(roc_aucs)):
			roc_auc = roc_aucs[i]
			title = titles[i]
			wandb_log[f"{title} ROC AUC"] = roc_auc
			titles[i] = f"{title} (area = {round(roc_auc, 2)})"

		# Add guessing as a line to the plot
		fprs += [np.array([0., 1.])]
		tprs += [np.array([0., 1.])]
		titles += ["Guessing (area = 0.50)"]

		# Add the roc curve to the wandb log
		plot_title = f"ROC AUCs at epoch {step}"
		wandb_log[plot_title] = wandb.plot.line_series(
			xs=fprs, ys=tprs, keys=titles, title=plot_title, xname="False Positive Rate"
		)
		wandb.log(wandb_log)

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
		# Choose either 8 or 16
		process_batch_size = 8

		# Extract the value that will be used in the attack dataset into separate variables
		features = torch.stack([torch.tensor(value[0][0]) for value in intercepted_data])
		labels = torch.stack([torch.tensor(value[0][1]) for value in intercepted_data])
		is_member = torch.stack([torch.tensor(value[1]) for value in intercepted_data])
		member_origins = torch.tensor([value[2] if value[2] else -1 for value in intercepted_data])
		num_classes = simulation.model_config.get_num_classes()

		# Translate the boolean to an integer
		is_member = is_member.int()

		# Reshape the metrics so they are 5D and can fit in a gradient component
		metrics = {
			key: [reshape_to_4d(torch.tensor(layer), True).detach().float() for layer in value] for key, value in
			metrics.items()
		}

		process_amount = math.ceil(len(features) / process_batch_size)

		def attack_dataset_generator():
			for i in range(process_amount):
				start = i * process_batch_size
				end = (i + 1) * process_batch_size

				# Stack the information of the shadow_model_simulation in tensors
				batch_features = features[start:end]
				batch_labels = labels[start:end]
				batch_is_member = is_member[start:end]
				batch_member_origins = member_origins[start:end]

				iteration_batch_size = len(batch_features)

				activation_values, gradients, loss = self._precompute_attack_features(
					batch_features, batch_labels, models, simulation
				)

				# One-hot encode the labels
				batch_labels = torch.nn.functional.one_hot(batch_labels, num_classes=num_classes)

				for j in range(iteration_batch_size):
					# Make label float as it is the input for another component
					label = batch_labels[j].float()

					# Gradient, activation values and loss values have a grad_fn detach the tensor from it
					gradient = [layer[j] for layer in gradients]
					activation_value = [layer[j].detach() for layer in activation_values]
					loss_value = loss[j].detach().unsqueeze(-1)

					is_value_member = batch_is_member[j].detach()
					value_origin = batch_member_origins[j].detach()

					yield (
						(gradient, activation_value, metrics, loss_value, label), is_value_member.float(), value_origin
					)

		# Helper variables for clarity
		dataset_size = len(intercepted_data)

		attack_dataset = attack_dataset_generator()
		dataset = GeneratorDataset(attack_dataset, dataset_size)

		batch_size = self._attack_simulation.batch_size
		if batch_size == -1:
			batch_size = dataset_size
		attack_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
		return attack_dataloader

	def _precompute_attack_features(self, features, labels, model_iterations, simulation):
		"""
		Gets the loss, activation functions and gradients for a list of parameters
		"""
		# Get the models from the parameter iterations
		template_model = simulation.model
		global_rounds = simulation.global_rounds + 1

		# Convert model iterations to state dicts
		keys = template_model.state_dict().keys()
		state_dicts = [{key: torch.tensor(model_iterations[j][i], requires_grad=True) for j, key in enumerate(keys)} for
					   i in range(global_rounds)]

		# Get the activation values
		activation_values = self._get_activation_values(state_dicts, features, simulation)
		predictions = activation_values[-1]
		losses = self._get_losses(predictions, labels, simulation)
		gradients = self._get_gradients(losses, state_dicts, simulation)
		losses = losses.T
		return activation_values, gradients, losses

	def _get_activation_values(self, model_state_dicts, features, simulation):
		# Helper variables
		template_model = simulation.model
		global_rounds = simulation.global_rounds + 1

		# Attach the hooks that will get the intermediate value of each layer
		activation = {}

		def get_activation(name):
			def hook(model, input, output):
				activation[name] = output.detach()

			return hook

		# Attack activation hook to all leaf modules
		hooks = [
			module.register_forward_hook(get_activation(name)) for name, module in template_model.named_modules()
			if len(list(module.children())) == 0
		]

		# Predict the features for each model iteration and capture the activation values from the dict
		activation_values = []
		for i in range(global_rounds):
			prediction = torch.func.functional_call(template_model, model_state_dicts[i], features)

			# Append prediction separately so grad_fn stays intact
			activation_values.append(list(activation.values())[:-1])
			activation_values[-1].append(prediction)

		# Remove the hooks
		[h.remove() for h in hooks]

		# Transpose the activations so that they are bundled per layer per batch sample instead of per model iteration

		# Transpose the activations so that they are bundled per layer instead of per model iteration
		activation_values = [torch.stack(layers, dim=1) for layers in zip(*activation_values)]
		return activation_values

	def _get_losses(self, predictions, label, simulation):
		"""
		Gets the losses of a list of predictions and a list of labels
		"""
		global_rounds = simulation.global_rounds + 1
		criterion = simulation.criterion
		criterion.reduction = "none"
		losses = torch.stack([criterion(predictions[:, i], label) for i in range(global_rounds)])
		return losses

	def _get_gradients(self, losses, model_state_dicts, simulation):
		gradients = []
		for i, state_dict in enumerate(model_state_dicts):
			model_gradients = []
			for loss in losses[i]:
				loss.backward(retain_graph=True)
				model_gradients.append((reshape_to_4d(state_dict[key].grad) for key in state_dict.keys()))
			gradients.append((torch.stack(layers) for layers in zip(*model_gradients)))

		# Transpose the gradients so that they are bundled per layer instead of per model iteration
		gradients = [torch.stack(layers, dim=1) for layers in zip(*gradients)]
		return gradients

	def _get_wandb_kwargs(self, attack_type, simulation_wandb_kwargs):
		simulation_wandb_kwargs["tags"][0] = f"Attack-{attack_type}"
		simulation_wandb_kwargs["config"] = {"simulation": simulation_wandb_kwargs["config"]}
		simulation_wandb_kwargs["config"]["attack"] = {
			"data_access": self._data_access,
			"message_access": self._message_access,
			"batch_size": self._attack_simulation.batch_size,
			"optimizer": self._attack_simulation.optimizer_name,
			**self._attack_simulation.optimizer_parameters
		}
		return simulation_wandb_kwargs


def reshape_to_4d(input_tensor: torch.Tensor, batched: bool = False) -> torch.Tensor:
	if batched:
		unsqueeze_dim = 2
		target_dim = 5
	else:
		unsqueeze_dim = 1
		target_dim = 4

	if input_tensor.ndim > target_dim:
		input_tensor = input_tensor.view(*input_tensor.shape[:(target_dim-1)], -1)
	elif input_tensor.ndim < target_dim:
		tensor_shape = input_tensor.shape
		target_shape = (
			*tensor_shape[:unsqueeze_dim], *(1,) * (target_dim - input_tensor.ndim), *tensor_shape[unsqueeze_dim:]
		)
		input_tensor = input_tensor.view(target_shape)
	return input_tensor

class GeneratorDataset(Dataset):
	def __init__(self, generator, length):
		self._generator = generator
		self._length = length
		self.data_cache = [None] * length
		self._initialize_cache()

	def _initialize_cache(self):
		gen = self._generator
		for i in tqdm(range(self._length), leave=True):
			self.data_cache[i] = next(gen)

	def __len__(self):
		return self._length

	def __getitem__(self, index):
		return self.data_cache[index]
