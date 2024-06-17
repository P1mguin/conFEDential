from typing import Dict, List


class ModelArchitecture:
	def __init__(self, components: Dict[str, bool], gradient_component: List[dict], fcn_component: List[dict],
				 encoder_component: List[dict]):
		self._components = components
		self._gradient_component = gradient_component
		self._fcn_component = fcn_component
		self._encoder_component = encoder_component

	def __str__(self):
		result = "ModelArchitecture:"
		result += "\n\tcomponents:"
		for key, value in self._components.items():
			result += f"\n\t\t{key}: {value}"
		result += "\n\tgradient_component:"
		for layer in self._gradient_component:
			result += "\n\t\t{}".format("\n\t\t\t".join([f"{key}: {value}" for key, value in layer.items()]))
		result += "\n\tfcn_component:"
		for layer in self._fcn_component:
			result += "\n\t\t{}".format("\n\t\t\t".join([f"{key}: {value}" for key, value in layer.items()]))
		result += "\n\tencoder_component:"
		for layer in self._encoder_component:
			result += "\n\t\t{}".format("\n\t\t\t".join([f"{key}: {value}" for key, value in layer.items()]))
		return result

	@staticmethod
	def from_dict(config: dict) -> 'ModelArchitecture':
		return ModelArchitecture(
			components=config['components'],
			gradient_component=config['gradient_component'],
			fcn_component=config['fcn_component'],
			encoder_component=config['encoder_component']
		)

	@property
	def fcn(self):
		return self._fcn_component

	@property
	def encoder(self):
		return self._encoder_component

	@property
	def gradient(self):
		return self._gradient_component

	@property
	def use_label(self):
		return self._components["label"]

	@property
	def use_loss(self):
		return self._components["loss"]

	@property
	def use_activation(self):
		return self._components["activation"]

	@property
	def use_gradient(self):
		return self._components["gradient"]

	@property
	def use_metrics(self):
		return self._components["metrics"]
