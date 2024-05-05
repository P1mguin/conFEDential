from __future__ import annotations

class Federation:
	def __init__(self, client_count: int, fraction_fit: float, global_rounds: int, local_rounds: int):
		self.client_count = client_count
		self.fraction_fit = fraction_fit
		self.global_rounds = global_rounds
		self.local_rounds = local_rounds

	def __str__(self):
		result = "Federation:"
		result += f"\n\tclient_count: {self.client_count}"
		result += f"\n\tfraction_fit: {self.fraction_fit}"
		result += f"\n\tglobal_rounds: {self.global_rounds}"
		result += f"\n\tlocal_rounds: {self.local_rounds}"
		return result

	def __repr__(self):
		result = "Federation("
		result += f"client_count={self.client_count}, "
		result += f"fraction_fit={self.fraction_fit}, "
		result += f"global_rounds={self.global_rounds}, "
		result += f"local_rounds={self.local_rounds}"
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