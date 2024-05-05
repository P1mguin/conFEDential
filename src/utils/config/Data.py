from __future__ import annotations

class Data:
	def __init__(self, dataset_name: str, batch_size: int, raw_preprocess_fn: str, splitter: dict | None=None) -> None:
		self.dataset_name = dataset_name
		self.batch_size = batch_size
		self.raw_preprocess_fn = raw_preprocess_fn
		self.is_split = splitter is not None
		if self.is_split:
			self.alpha = float(splitter['alpha'])
			self.percent_non_iid = float(splitter['percent_non_iid'])

	def __str__(self):
		result = "Data:"
		result += f"\n\tdataset_name: {self.dataset_name}"
		result += f"\n\tbatch_size: {self.batch_size}"
		result += "\n\tpreprocess_fn:\n\t\t{}".format("\n\t\t".join(self.raw_preprocess_fn.split("\n")[:-1]))
		if self.is_split:
			result += "\n\tsplitter:"
			result += f"\n\t\talpha: {self.alpha}"
			result += f"\n\t\tpercent_non_iid: {self.percent_non_iid}"
		return result

	def __repr__(self):
		result = "Data("
		result += f"dataset_name={self.dataset_name}, "
		result += f"batch_size={self.batch_size}"
		if self.is_split:
			result += f", alpha={self.alpha}, "
			result += f"percent_non_iid={self.percent_non_iid}"
		result += ")"
		return result

	@staticmethod
	def from_dict(config: dict) -> Data:
		return Data(
			dataset_name=config['dataset_name'],
			batch_size=config['batch_size'],
			raw_preprocess_fn=config['preprocess_fn'],
			splitter=config.get('splitter', None)
		)