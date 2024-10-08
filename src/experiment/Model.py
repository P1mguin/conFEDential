from typing import Iterator, List

import flwr as fl
import torch
import torch.nn as nn

from src import training, utils


class Model:
	def __init__(
			self,
			cache_root: str,
			optimizer_name: str,
			model_name: str,
			criterion_name: str,
			model_architecture: List[dict] | dict,
			optimizer_parameters: dict | None = None
	):
		if optimizer_parameters is None:
			optimizer_parameters = {}
		self._optimizer_name = optimizer_name
		self._model_name = model_name
		self._criterion_name = criterion_name
		self._optimizer_parameters = optimizer_parameters
		self._model_architecture = model_architecture
		self._cache_root = cache_root

		self._model = None
		self._prepare_model()
		self._criterion = None
		self._prepare_criterion()
		self._learning_method = None
		self._prepare_learning_method()

	def __str__(self):
		result = "Model:"
		result += f"\n\toptimizer_name: {self._optimizer_name}"
		result += f"\n\tmodel_name: {self._model_name}"
		result += f"\n\tcriterion_name: {self._criterion_name}"
		result += "\n\toptimizer_parameters: \n\t\t{}".format(
			"\n\t\t".join([f"{key}: {value}" for key, value in self._optimizer_parameters.items()]))
		result += "\n\tmodel_architecture:"
		is_from_hub = isinstance(self._model_architecture, dict)
		if is_from_hub:
			result += "\n\t\t{}".format(
				"\n\t\t\t".join([f"{key}: {value}" for key, value in self._model_architecture.items()])
			)
		else:
			for layer in self._model_architecture:
				result += "\n\t\t{}".format("\n\t\t\t".join([f"{key}: {value}" for key, value in layer.items()]))
		return result

	def __repr__(self):
		result = "Model("
		result += f"optimizer={self._optimizer_name}("
		result += "{}), ".format(", ".join([f"{key}={value}" for key, value in self._optimizer_parameters.items()]))
		result += f"model_name={self._model_name}, "
		result += f"criterion_name={self._criterion_name}"
		result += ")"
		return result

	@staticmethod
	def from_dict(config: dict, cache_root: str) -> 'Model':
		return Model(
			cache_root=cache_root,
			optimizer_name=config['optimizer_name'],
			model_name=config['model_name'],
			criterion_name=config['criterion_name'],
			optimizer_parameters=config['optimizer_parameters'],
			model_architecture=config['model_architecture']
		)

	@property
	def model(self):
		return self._model

	@property
	def criterion(self):
		return self._criterion()

	@property
	def learning_method(self):
		return self._learning_method

	@property
	def model_name(self):
		return self._model_name

	@property
	def optimizer_name(self):
		return self._optimizer_name

	@property
	def optimizer_parameters(self):
		return self._optimizer_parameters

	def get_optimizer(self, parameters: Iterator[nn.Parameter]) -> torch.optim.Optimizer:
		return self._learning_method.get_optimizer(parameters)

	def get_trainable_layer_indices(self):
		return set(int(name.split(".")[1]) for name, _ in self.model.named_parameters())

	def get_initial_parameters(self):
		model = self.model
		learning_method_class = getattr(training.learning_methods, self._optimizer_name)
		# The learning method may require some specific initial parameters which can be implemented as a class method
		if hasattr(learning_method_class, "get_initial_parameters"):
			return learning_method_class.get_initial_parameters(model)
		else:
			weights = training.get_weights(model)
			return fl.common.ndarrays_to_parameters(weights)

	def get_num_classes(self):
		return next(
			layer.out_features for layer in reversed(list(self._model.modules())) if hasattr(layer, "out_features")
		)

	def _prepare_model(self):
		is_from_hub = isinstance(self._model_architecture, dict)
		if is_from_hub:
			model = self._load_model_from_hub()
		else:
			model = self._load_model_from_layers()

		self._model = model

	def _load_model_from_layers(self):
		# Construct a list of nn Modules
		layers = []
		for layer in self._model_architecture:
			layer_type = getattr(nn, layer["type"])
			parameters = {key: value for key, value in list(layer.items())[1:]}
			layer = layer_type(**parameters)
			layers.append(layer)

		# Convert that to a model and return
		model = utils.get_net_class_from_layers(layers)
		return model()

	def _load_model_from_hub(self):
		model_path = self._get_model_cache_path()

		def get_model():
			base_model = torch.load(model_path)
			out_features = self._model_architecture["out_features"]
			in_features = base_model.fc.in_features
			base_model.fc = nn.Linear(in_features, out_features)
			return base_model

		return get_model()

	def _get_model_cache_path(self):
		repo = self._model_architecture["repo_or_dir"]
		model_name = self._model_architecture["model"]

		return f"{self._cache_root}model_architectures/{repo.replace('/', '')}_{model_name}.pth"

	def _prepare_criterion(self):
		self._criterion = getattr(nn, self._criterion_name)

	def _prepare_learning_method(self):
		learning_method_class = getattr(training.learning_methods, self._optimizer_name)
		learning_method = learning_method_class(**self._optimizer_parameters)
		self._learning_method = learning_method

		initial_parameters = self.get_initial_parameters()
		self._learning_method.set_parameters(initial_parameters)
