from typing import List


class ModelArchitecture:
	def __init__(self, gradient_component: List[dict], fcn_component: List[dict], encoder_component: List[dict]):
		self._gradient_component = gradient_component
		self._fcn_component = fcn_component
		self._encoder_component = encoder_component

	def __str__(self):
		result = "ModelArchitecture:"
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
