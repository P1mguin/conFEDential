from __future__ import annotations

from typing import List

from .AttackSimulation import AttackSimulation


class Attack:
	def __init__(
			self,
			is_targeted: bool,
			data_access: str,
			message_access: str,
			shadow_model_amount: int,
			targets: int | None = None,
			attack_simulation: AttackSimulation | None = None
	):
		self._is_targeted = is_targeted
		self._targets = targets
		self._data_access = data_access
		self._message_access = message_access
		self._shadow_model_amount = shadow_model_amount
		self._attack_simulation = attack_simulation

		self._client_id = None

	def __str__(self):
		result = "Attack:"
		result += f"\n\tis_targeted: {self._is_targeted}"
		result += f"\n\tdata_access: {self._data_access}"
		result += f"\n\tmessage_access: {self._message_access}"
		result += f"\n\tshadow_model_amount: {self._shadow_model_amount}"
		if self._is_targeted:
			result += f"\n\ttargets: {self._targets}"
		else:
			result += "\n\t{}".format("\n\t".join(str(self._attack_simulation).split("\n")))
		return result

	def __repr__(self):
		result = "Attack("
		result += f"is_targeted={self._is_targeted}, "
		result += f"data_access={self._data_access}, "
		result += f"message_access={self._message_access}, "
		result += f"shadow_model_amount={self._shadow_model_amount}"
		if self._is_targeted:
			result += f", targets={self._targets}"
		else:
			result += f", {repr(self._attack_simulation)}"
		result += ")"
		return result

	@staticmethod
	def from_dict(config: dict) -> Attack:
		is_targeted = config['is_targeted']
		if is_targeted:
			attack_simulation = None
			targets = config['targets']
		else:
			targets = None
			attack_simulation = AttackSimulation.from_dict(config['attack_simulation'])

		return Attack(
			is_targeted=is_targeted,
			data_access=config['data_access'],
			message_access=config['message_access'],
			shadow_model_amount=config['shadow_model_amount'],
			attack_simulation=attack_simulation,
			targets=targets
		)

	@property
	def is_targeted(self) -> bool:
		return self._is_targeted

	@property
	def client_id(self) -> int:
		return self._client_id

	@client_id.setter
	def client_id(self, client_id: int):
		self._client_id = client_id

	@property
	def data_access(self) -> str:
		return self._data_access

	@property
	def targets(self) -> int | None:
		return self._targets

	@property
	def shadow_model_amount(self) -> int:
		return self._shadow_model_amount

	def get_data_access_indices(self, client_count) -> List[int]:
		if self._data_access == "client":
			return [self._client_id]
		elif self._data_access == "all":
			return list(range(client_count))
