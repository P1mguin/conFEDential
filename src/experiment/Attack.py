import math
import random
from logging import DEBUG, INFO
from typing import List

import h5py
import more_itertools
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import wandb
from flwr.common import log
from rdp import rdp
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.training as training
import src.utils as utils
from src.experiment import AttackSimulation


class Attack:
	def __init__(
			self,
			data_access: float,
			message_access: str,
			aggregate_access: List[int] | int | float,
			repetitions: int,
			attack_simulation=None
	):
		self._data_access = data_access
		self._message_access = message_access
		self._aggregate_access = aggregate_access
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
		result += f"\n\taggregate_access: {self._aggregate_access}"
		result += f"\n\trepetitions: {self._repetitions}"
		result += "\n\t{}".format("\n\t".join(str(self._attack_simulation).split("\n")))
		return result

	def __repr__(self):
		result = "Attack("
		result += f"data_access={self._data_access}, "
		result += f"message_access={self._message_access}, "
		result += f"aggregate_access={self._aggregate_access}, "
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
			aggregate_access=config['aggregate_access'],
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
	def message_access(self) -> str:
		return self._message_access

	@property
	def repetitions(self) -> int:
		return self._repetitions

	@property
	def attack_simulation(self):
		return self._attack_simulation

	def train_membership_inference_net(
			self,
			attack_model,
			train_loader,
			validation_loader,
			test_loader,
			wandb_kwargs
	):
		wandb_kwargs = self._get_wandb_kwargs("membership-inference", wandb_kwargs)
		wandb.init(**wandb_kwargs)

		# Get training algorithm variables and set model in training mode
		attack_model = attack_model.to(training.DEVICE)
		criterion = nn.MSELoss()
		optimizer = self._attack_simulation.get_optimizer(attack_model.parameters())

		# Set initial parameters and get initial performance
		test_roc_auc, test_fpr, test_tpr, test_loss, test_accuracy = self._test_model(attack_model, test_loader)
		val_roc_auc, val_fpr, val_tpr, val_loss, val_accuracy = self._test_model(attack_model, validation_loader)

		# Log the initial performance
		log(
			INFO,
			f"Initial performance: validation auc {val_roc_auc}, test auc {test_roc_auc},"
			f" validation loss {val_loss}, test loss {test_loss},"
			f" validation accuracy {val_accuracy}, test accuracy {test_accuracy}"
		)
		if len(test_loader.dataset) == 1:
			self._log_aucs(
				[val_roc_auc],
				[val_fpr],
				[val_tpr],
				[val_loss, test_loss],
				[val_accuracy, test_accuracy],
				["Validation", "Test"],
				log_roc=False,
				step=-1
			)
		else:
			self._log_aucs(
				[val_roc_auc, test_roc_auc],
				[val_fpr, test_fpr],
				[val_tpr, test_tpr],
				[val_loss, test_loss],
				[val_accuracy, test_accuracy],
				["Validation", "Test"],
				log_roc=False,
				step=-1
			)

		# Training loop with early stopping over the average loss of the last 5 rounds with patience of 10 rounds
		patience = 10
		patience_counter = 0
		relative_tolerance = 1e-3
		average_over = 5
		losses = [val_loss]
		average_loss = val_loss

		i = 0
		breaking = False
		while True:
			attack_model.train()
			previous_average_loss = average_loss
			predictions = torch.Tensor().to(training.DEVICE)
			is_members = torch.Tensor().to(training.DEVICE)
			for (
					gradient,
					activation_value,
					metrics,
					loss_value,
					label
			), is_value_member, _ in tqdm(train_loader, leave=True):
				optimizer.zero_grad()
				prediction = attack_model(gradient, activation_value, metrics, loss_value, label)

				# Delete memory heavy objects
				del gradient, activation_value, metrics, loss_value, label

				# Do the backwards step
				loss = criterion(prediction, is_value_member)
				loss.backward()
				optimizer.step()

				# Log the prediction for later performance evaluation
				predictions = torch.cat((predictions, prediction.detach()))
				is_members = torch.cat((is_members, is_value_member.detach()))
				del prediction

			# Get the performance after the epoch
			train_fpr, train_tpr, _ = roc_curve(is_members.cpu(), predictions.cpu())
			train_roc_auc = auc(train_fpr, train_tpr)

			# Downsample the fpr and tpr
			line = np.stack([train_fpr, train_tpr], axis=1)
			lines = rdp(line, epsilon=0.0001)
			train_fpr, train_tpr = lines[:, 0], lines[:, 1]

			# Get the train loss and accuracy
			train_loss = criterion(predictions, is_members)
			train_accuracy = (predictions.round() == is_members).sum().item() / is_members.size(0)

			test_roc_auc, test_fpr, test_tpr, test_loss, test_accuracy = self._test_model(attack_model, test_loader)
			val_roc_auc, val_fpr, val_tpr, val_loss, val_accuracy = self._test_model(attack_model, validation_loader)

			# Get the average loss over the last few values
			losses.append(val_loss)
			if len(losses) > average_over:
				losses.pop(0)
			average_loss = sum(losses) / len(losses)

			# Early stopping
			loss_decrease = -(average_loss - previous_average_loss) / previous_average_loss
			log(DEBUG, f"Loss decrease: {loss_decrease}")

			if loss_decrease < relative_tolerance:
				patience_counter += 1
				if patience_counter >= patience:
					log(
						INFO,
						"Early stopping at round %s, loss %s",
						i,
						val_loss
					)
					breaking = True
			else:
				patience_counter = 0

			# Log the performance
			log(
				INFO,
				f"Epoch {i}: train auc {train_roc_auc}, validation auc {val_roc_auc}, test auc {test_roc_auc},"
				f" train loss {train_loss}, validation loss {val_loss}, test loss {test_loss},"
				f" train accuracy {train_accuracy}, validation accuracy {val_accuracy}, test accuracy {test_accuracy}"
			)
			if len(test_loader.dataset) == 1:
				self._log_aucs(
					[train_roc_auc, val_roc_auc],
					[train_fpr, val_fpr],
					[train_tpr, val_tpr],
					[train_loss, val_loss, test_loss],
					[train_accuracy, val_accuracy, test_accuracy],
					["Train", "Validation", "Test"],
					log_roc=breaking,
					step=i
				)
			else:
				self._log_aucs(
					[train_roc_auc, val_roc_auc, test_roc_auc],
					[train_fpr, val_fpr, test_fpr],
					[train_tpr, val_tpr, test_tpr],
					[train_loss, val_loss, test_loss],
					[train_accuracy, val_accuracy, test_accuracy],
					["Train", "Validation", "Test"],
					log_roc=breaking,
					step=i
				)

			if breaking:
				break

			i += 1
		wandb.finish()
		return train_roc_auc

	def _test_model(self, model, dataloader):
		# Set the model on the right device and put it in testing mode
		model = model.to(training.DEVICE)
		model.eval()

		criterion = nn.MSELoss()

		predictions = torch.Tensor().to(training.DEVICE)
		is_members = torch.Tensor().to(training.DEVICE)
		for (
				gradient,
				activation_value,
				metrics,
				loss_value,
				label
		), is_value_member, _ in tqdm(dataloader, leave=True):
			with torch.no_grad():
				prediction = model(gradient, activation_value, metrics, loss_value, label)
				predictions = torch.cat((predictions, prediction))
				is_members = torch.cat((is_members, is_value_member))

		# Compute the roc auc
		fpr, tpr, _ = roc_curve(is_members.cpu(), predictions.cpu())
		roc_auc = auc(fpr, tpr)

		# Get the loss
		loss = criterion(predictions, is_members)

		# Get the accuracy
		accuracy = (predictions.round() == is_members).sum().item() / is_members.size(0)

		return roc_auc, fpr, tpr, loss, accuracy

	def _log_aucs(self, roc_aucs, fprs, tprs, losses, accuracies, titles, log_roc: bool, step: int):
		wandb_log = {}

		# Add the losses to the wandb log
		for i in range(len(losses)):
			loss = losses[i]
			title = titles[i]
			wandb_log[f"{title} loss"] = loss

		# Add the accuracies to the wandb log
		for i in range(len(accuracies)):
			accuracy = accuracies[i]
			title = titles[i]
			wandb_log[f"{title} accuracy"] = accuracy

		# Add the auc to the wandb log
		for i in range(len(roc_aucs)):
			roc_auc = roc_aucs[i]
			title = titles[i]
			wandb_log[f"{title} ROC AUC"] = roc_auc
			titles[i] = f"{title} (area = {round(roc_auc, 2)})"

		if log_roc:
			# Remove the titles that do not have a roc auc
			titles = titles[:len(roc_aucs)]

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

	def get_aggregate_access_indices(self, aggregate_path):
		# Get how many rounds there were (including the initial model)
		with h5py.File(aggregate_path, 'r') as hf:
			global_rounds = hf["0"].shape[0]

		if isinstance(self._aggregate_access, int):
			# Return the last n rounds
			aggregate_access_indices = list(range(global_rounds - self._aggregate_access, global_rounds))
		elif isinstance(self._aggregate_access, float):
			# Return n fraction of the rounds counting from the back
			aggregate_access_indices = list(
				range(global_rounds - int(global_rounds * self._aggregate_access), global_rounds))
		elif isinstance(self._aggregate_access, list):
			# Return the specified rounds
			aggregate_access_indices = self._aggregate_access
		else:
			raise ValueError("Aggregate access should be an int, float or list of ints")

		return aggregate_access_indices

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
			non_member_loader = simulation.non_member_loader
			self._target_client = None
			self._target = random.choice([*test_loader.dataset, *non_member_loader.dataset])

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
		process_batch_size = 16

		# Extract the value that will be used in the attack dataset into separate variables
		features = torch.stack([torch.tensor(value[0][0], device=training.DEVICE) for value in intercepted_data])
		labels = torch.stack([torch.tensor(value[0][1], device=training.DEVICE) for value in intercepted_data])
		is_member = torch.stack([torch.tensor(value[1], device=training.DEVICE) for value in intercepted_data])
		member_origins = torch.tensor([value[2] if value[2] else -1 for value in intercepted_data], device=training.DEVICE)
		num_classes = simulation.model_config.get_num_classes()

		# Translate the boolean to an integer
		is_member = is_member.int()

		# Convert the elements in the metrics to a tensor
		if self.attack_simulation.model_architecture.use_metrics:
			metrics = {
				key: [torch.tensor(layer, device=training.DEVICE).detach().float() for layer in value] for key, value in
				metrics.items()
			}
		else:
			metrics = {}

		process_amount = math.ceil(len(features) / process_batch_size)

		def attack_dataset_generator(random_seed=None):
			if random_seed is not None:
				random.seed(random_seed)

			indices = set(range(len(features)))
			for i in range(process_amount):
				try:
					batch_indices = random.sample(list(indices), process_batch_size)
				except ValueError:
					batch_indices = list(indices)

				indices = indices - set(batch_indices)

				# Stack the information of the shadow_model_simulation in tensors
				batch_features = features[batch_indices]
				batch_labels = labels[batch_indices]
				batch_is_member = is_member[batch_indices]
				batch_member_origins = member_origins[batch_indices]

				iteration_batch_size = len(batch_indices)

				activation_values, gradients, loss, metric_update = self._precompute_attack_features(
					batch_features, batch_labels, models, metrics, simulation
				)

				# One-hot encode the labels
				batch_labels = torch.nn.functional.one_hot(batch_labels, num_classes=num_classes)

				for j in range(iteration_batch_size):
					# Make label float as it is the input for another component
					label = batch_labels[j].float()

					# Gradient, activation values and loss values have a grad_fn detach the tensor from it
					gradient = [layer[j].detach() for layer in gradients]
					activation_value = [layer[j].detach() for layer in activation_values]
					metric_update_value = {
						key: [layer[j].detach() for layer in value] for key, value in metric_update.items()
					}
					if self.attack_simulation.model_architecture.use_loss:
						loss_value = loss[j].detach().unsqueeze(-1)
					else:
						loss_value = torch.tensor(-1)

					is_value_member = batch_is_member[j].detach()
					value_origin = batch_member_origins[j].detach()

					yield (
						(gradient, activation_value, metric_update_value, loss_value, label),
						is_value_member.float(),
						value_origin
					)

		# Helper variables for clarity
		dataset_size = len(intercepted_data)

		dataset = IterableDataset(attack_dataset_generator, dataset_size)

		batch_size = self._attack_simulation.batch_size
		if batch_size < 0:
			batch_size = dataset_size // abs(batch_size)

		attack_dataloader = DataLoader(dataset, batch_size=batch_size)
		return attack_dataloader

	def _precompute_attack_features(self, features, labels, model_iterations, metrics, simulation):
		"""
		Gets the loss, activation functions and gradients for a list of parameters
		"""
		# Get the models from the parameter iterations
		template_model = simulation.model
		intercepted_aggregate_round_count = model_iterations[0].shape[0]
		model_architecture = self.attack_simulation.model_architecture

		# Convert model iterations to state dicts
		keys = template_model.state_dict().keys()
		parameter_keys = set(name for name, _ in template_model.named_parameters())
		state_dicts = [{key: torch.tensor(
			model_iterations[j][i],
			dtype=torch.float32,
			requires_grad=(key in parameter_keys),
			device=training.DEVICE
		) for j, key in enumerate(keys)} for i in range(intercepted_aggregate_round_count)]

		# Get the activation values
		activation_values = self._get_activation_values(state_dicts, features, simulation)
		predictions = activation_values[-1]

		gradients, losses, metric_update = [], [], {}
		if model_architecture.use_loss or model_architecture.use_gradient or model_architecture.use_metrics:
			losses = self._get_losses(predictions, labels, simulation)
			if model_architecture.use_gradient:
				gradients = self._get_gradients(losses, state_dicts, simulation)
			if model_architecture.use_loss:
				losses = losses.T
			if model_architecture.use_metrics:
				metric_update = simulation.learning_method.compute_metric_updates(
					state_dicts=state_dicts,
					metrics=metrics,
					features=features,
					labels=labels,
					simulation=simulation
				)

		return activation_values, gradients, losses, metric_update

	def _get_activation_values(self, model_state_dicts, features, simulation):
		# Helper variables
		template_model = simulation.model
		intercepted_aggregate_round_count = len(model_state_dicts)

		# Attach the hooks that will get the intermediate value of each layer
		if self.attack_simulation.model_architecture.use_activation:
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

		# Some layers may require batch to be bigger than 1, for instance BatchNorm. Duplicating it has no effect since the
		# tracked variables are not used in further computations
		is_singular_batch = features.size(0) == 1
		if is_singular_batch:
			features = torch.cat([features, features])

		for i in range(intercepted_aggregate_round_count):
			prediction = torch.func.functional_call(template_model, model_state_dicts[i], features)

			# Append prediction separately so grad_fn stays intact
			if self.attack_simulation.model_architecture.use_activation:
				activation_values.append(list(activation.values())[:-1])
				activation_values[-1].append(prediction)
			else:
				activation_values.append([prediction])

		# Remove the hooks
		if self.attack_simulation.model_architecture.use_activation:
			[h.remove() for h in hooks]

		if is_singular_batch:
			activation_values = [
				[x[0].unsqueeze(0) for x in global_round_values] for global_round_values in activation_values
			]

		# Transpose the activations so that they are bundled per layer instead of per model iteration
		activation_values = [torch.stack(layers, dim=1) for layers in zip(*activation_values)]
		return activation_values

	def _get_losses(self, predictions, label, simulation):
		"""
		Gets the losses of a list of predictions and a list of labels
		"""
		criterion = simulation.criterion
		criterion.reduction = "none"
		losses = torch.stack([criterion(predictions[:, i], label) for i in range(predictions.shape[1])])
		return losses

	def _get_gradients(self, losses, model_state_dicts, simulation):
		gradients = []
		parameter_keys = [name for name, _ in simulation.model.named_parameters()]
		for i, state_dict in enumerate(model_state_dicts):
			model_gradients = []
			for loss in losses[i]:
				loss.backward(retain_graph=True)
				model_gradients.append((utils.reshape_to_4d(state_dict[key].grad) for key in parameter_keys))
			gradients.append((torch.stack(layers) for layers in zip(*model_gradients)))

		# Transpose the gradients so that they are bundled per layer instead of per model iteration
		gradients = [torch.stack(layers, dim=1) for layers in zip(*gradients)]
		return gradients

	def _get_wandb_kwargs(self, attack_type, simulation_wandb_kwargs):
		simulation_wandb_kwargs["tags"][0] = f"Attack-{attack_type}"
		simulation_wandb_kwargs["config"] = {"simulation": simulation_wandb_kwargs["config"]}
		model_config = self._attack_simulation.model_architecture
		simulation_wandb_kwargs["config"]["attack"] = {
			"data_access": self._data_access,
			"message_access": self._message_access,
			"aggregate_access": self._aggregate_access,
			"batch_size": self._attack_simulation.batch_size,
			"optimizer": self._attack_simulation.optimizer_name,
			"label": model_config.use_label,
			"loss": model_config.use_loss,
			"activation": model_config.use_activation,
			"gradient": model_config.use_gradient,
			"metrics": model_config.use_metrics,
			**self._attack_simulation.optimizer_parameters
		}
		return simulation_wandb_kwargs


class IterableDataset(data.IterableDataset):
	def __init__(self, generator_fn, length):
		self._generator_fn = generator_fn
		self._generator = None
		self.reset_generator()

		self._length = length

	def __iter__(self):
		# See if an item is available from the generator, if not reset generator
		try:
			self._generator.peek()
		except StopIteration:
			self.reset_generator()

		return iter(self._generator)

	def __len__(self):
		return self._length

	def reset_generator(self):
		self._generator = more_itertools.peekable(self._generator_fn())
