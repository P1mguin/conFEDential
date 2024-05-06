from __future__ import annotations

class Federation:
	def __init__(self, client_count: int, fraction_fit: float, global_rounds: int, local_rounds: int):
		self._client_count = client_count
		self._fraction_fit = fraction_fit
		self._global_rounds = global_rounds
		self._local_rounds = local_rounds

	def __str__(self):
		result = "Federation:"
		result += f"\n\tclient_count: {self._client_count}"
		result += f"\n\tfraction_fit: {self._fraction_fit}"
		result += f"\n\tglobal_rounds: {self._global_rounds}"
		result += f"\n\tlocal_rounds: {self._local_rounds}"
		return result

	def __repr__(self):
		result = "Federation("
		result += f"client_count={self._client_count}, "
		result += f"fraction_fit={self._fraction_fit}, "
		result += f"global_rounds={self._global_rounds}, "
		result += f"local_rounds={self._local_rounds}"
		result += ")"
		return result

	@staticmethod
	def from_dict(config: dict) -> Federation:
		return Federation(
			client_count=config['client_count'],
			fraction_fit=config['fraction_fit'],
			global_rounds=config['global_rounds'],
			local_rounds=config['local_rounds']
		)

	@property
	def client_count(self):
		return self._client_count

	@property
	def fraction_fit(self):
		return self._fraction_fit

	@property
	def global_rounds(self):
		return self._global_rounds

	@property
	def local_rounds(self):
		return self._local_rounds
