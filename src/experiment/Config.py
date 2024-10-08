import os
import random
from logging import INFO

import torch
import yaml
from flwr.common import log

from src import utils
from src.attacks import MembershipNet
from src.experiment import Attack, Simulation


class Config:
	def __init__(self, simulation, attack):
		self._simulation = simulation
		self._attack = attack

		if self._attack.data_access == "client":
			# Set the client id we are simulating the attack from
			self.attack.client_id = random.randint(0, self.simulation.client_count - 1)

	def __str__(self):
		result = "Config:"
		result += "\n\t{}".format("\n\t".join(str(self._simulation).split("\n")))
		result += "\n\t{}".format("\n\t".join(str(self._attack).split("\n")))
		return result

	def __repr__(self):
		result = "Config("
		result += f"{repr(self._simulation)}, "
		result += f"{repr(self._attack)}"
		result += ")"
		return result

	@staticmethod
	def from_yaml_file(yaml_file: str, cache_root) -> 'Config':
		with open(yaml_file, "r") as f:
			config = yaml.safe_load(f)

		return Config.from_dict(config, cache_root)

	@staticmethod
	def from_dict(config: dict, cache_root: str) -> 'Config':
		return Config(
			simulation=Simulation.from_dict(config['simulation'], cache_root),
			attack=Attack.from_dict(config['attack'])
		)

	@property
	def simulation(self):
		return self._simulation

	@property
	def attack(self):
		return self._attack

	def run_simulation(
			self,
			concurrent_clients: int,
			memory: int | None,
			num_cpus: int,
			num_gpus: int,
			is_ray_initialised: bool,
			is_online: bool,
			is_capturing: bool,
			run_name: str,
	):
		log(INFO, "Starting conFEDential simulation")
		# If nothing has been captured for this simulation, capture the simulation
		federation_simulation_capture_directory = self.simulation.get_capture_directory()
		capture_hash = federation_simulation_capture_directory.split("/")[-2]
		if not os.path.exists(f"{federation_simulation_capture_directory}aggregates"):
			log(INFO,
				f"No previous federated learning simulation found with hash {capture_hash}, starting training simulation...")
			self.simulation.simulate_federation(
				concurrent_clients, memory, num_cpus, num_gpus, is_ray_initialised, is_capturing, is_online, run_name
			)
		else:
			log(INFO,
				f"Found previous federated learning simulation with hash {capture_hash}, continuing to attack simulation...")

		for i in range(self.attack.repetitions):
			log(INFO, f"Starting {i + 1}th repetition of the attack simulation")
			j = 0
			while True:
				j += 1
				log(INFO, f"Starting {j}th attempt for stable attack")
				# Clean the variables of the attack
				self.attack.reset_variables()

				# For each intercepted datapoint, get their gradients, activation functions, loss
				# Add fraction eval for debugging purposes
				fraction_test = 1.0
				fraction_train = 0.85
				(
					train_dataset,
					validation_dataset,
					test_dataset
				) = self._get_attack_datasets(fraction_train, fraction_test)

				# Uncomment to analyse the Kullback-Leibler and Jensen-Shannon divergence of each variable
				# utils.do_analyses(validation_dataset)

				# Uncomment to visualize the difference between members and non-members
				# utils.visualize_loss_difference(validation_dataset, visualize_per_class=True)
				# utils.visualize_loss_difference(validation_dataset, log_scale=True, visualize_per_class=True)
				# utils.visualize_confidence_difference(validation_dataset, visualize_per_class=True)
				# utils.visualize_logit_difference(validation_dataset, visualize_per_class=True)

				# Get the template model and train it
				membership_net = self._construct_membership_net(train_dataset)

				wandb_kwargs = self.simulation.get_wandb_kwargs(run_name)
				mode = "online" if is_online else "offline"
				wandb_kwargs = {**wandb_kwargs, "mode": mode}
				train_roc_auc = self.attack.train_membership_inference_net(
					membership_net, train_dataset, validation_dataset, test_dataset, wandb_kwargs
				)
				if round(train_roc_auc, 1) != 0.5 or j == 9:
					break

	def _construct_membership_net(self, attack_dataset):
		(
			gradient,
			activation_value,
			metrics,
			loss_value,
			label
		), is_value_member, value_origin = next(iter(attack_dataset))
		attack_dataset.dataset.reset_generator()

		# Construct the attack model from all the shapes
		gradient_shapes = [gradient.shape[1:] for gradient in gradient]
		activation_shapes = [activation.shape[1:] for activation in activation_value]
		metrics_shapes = {key: [layer.shape[1:] for layer in metric] for key, metric in metrics.items()}
		if self.attack.attack_simulation.model_architecture.use_label:
			label_shape = label.shape[1:]
		else:
			label_shape = torch.empty(0, )

		# Get the attack model
		membership_net = MembershipNet(self, gradient_shapes, activation_shapes, metrics_shapes, label_shape)
		return membership_net

	def _get_attack_datasets(self, fraction_train: float = 0.85, fraction_test: float = 1.0):
		# Get the captured server aggregates
		aggregate_path = f"{self.simulation.get_capture_directory()}aggregates/aggregates.hdf5"
		aggregate_access_indices = self.attack.get_aggregate_access_indices(aggregate_path)
		message_access = self.attack.message_access
		aggregated_models, aggregated_metrics = self.simulation.get_server_aggregates(aggregate_access_indices,
																					  message_access)

		# Get the intercepted samples
		intercepted_data, remaining_data = self._get_intercepted_samples(fraction_test)

		# Split the intercepted data in training and validation
		training_length = int(len(intercepted_data) * fraction_train)
		intercepted_indices = set(range(len(intercepted_data)))
		training_indices = set(random.sample(list(intercepted_indices), training_length))
		validation_indices = intercepted_indices - training_indices
		training_data = [intercepted_data[i] for i in training_indices]
		validation_data = [intercepted_data[i] for i in validation_indices]

		# Get the datasets
		training_dataset = self.attack.get_membership_inference_attack_dataset(
			aggregated_models,
			aggregated_metrics,
			training_data,
			self.simulation
		)

		validation_dataset = self.attack.get_membership_inference_attack_dataset(
			aggregated_models,
			aggregated_metrics,
			validation_data,
			self.simulation
		)

		test_dataset = self.attack.get_membership_inference_attack_dataset(
			aggregated_models,
			aggregated_metrics,
			remaining_data,
			self.simulation
		)

		return training_dataset, validation_dataset, test_dataset

	def _get_intercepted_samples(self, fraction_eval: float = 1.0):
		# Get a target from all possible data
		target, is_target_member, target_origin = self.attack.get_target_sample(self.simulation)

		# Combine the data in one big dataset, annotated with the client origin and if it is trained on
		train_loaders = self.simulation.train_loaders
		test_loader = self.simulation.test_loader
		non_member_loader = self.simulation.non_member_loader
		training_data = [
			(sample, True, client_id) for client_id, train_loader in enumerate(train_loaders)
			for sample in train_loader.dataset if sample is not target
		]
		testing_data = [(sample, False, None) for sample in test_loader.dataset if sample is not target]
		non_member_data = [(sample, False, None) for sample in non_member_loader.dataset if sample is not target]
		data = [*training_data, *testing_data, *non_member_data]
		num_elements = round(self.attack.data_access * len(data))

		intercepted_indices = set(random.sample(range(len(data)), num_elements))
		remaining_indices = set(range(len(data))) - intercepted_indices
		remaining_indices = random.sample(list(remaining_indices), round(fraction_eval * len(remaining_indices)))

		intercepted_samples = [data[i] for i in intercepted_indices]
		remaining_samples = [data[i] for i in remaining_indices]
		remaining_samples += [(target, is_target_member, target_origin)]

		return intercepted_samples, remaining_samples
