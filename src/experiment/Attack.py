from __future__ import annotations

from .AttackSimulation import AttackSimulation


class Attack:
	def __init__(
			self,
			is_targeted: bool,
			data_access: str,
			message_access: str,
			shadow_model_amount: int,
			attack_simulation: AttackSimulation | None = None
	):
		self._is_targeted = is_targeted
		self._data_access = data_access
		self._message_access = message_access
		self._shadow_model_amount = shadow_model_amount
		self._attack_simulation = attack_simulation

	def __str__(self):
		result = "Attack:"
		result += f"\n\tis_targeted: {self._is_targeted}"
		result += f"\n\tdata_access: {self._data_access}"
		result += f"\n\tmessage_access: {self._message_access}"
		result += f"\n\tshadow_model_amount: {self._shadow_model_amount}"
		if not self._is_targeted:
			result += "\n\t{}".format("\n\t".join(str(self._attack_simulation).split("\n")))

		return result

	def __repr__(self):
		result = "Attack("
		result += f"is_targeted={self._is_targeted}, "
		result += f"data_access={self._data_access}, "
		result += f"message_access={self._message_access}, "
		result += f"shadow_model_amount={self._shadow_model_amount}"
		if not self._is_targeted:
			result += f", {repr(self._attack_simulation)}"
		result += ")"
		return result

	@staticmethod
	def from_dict(config: dict) -> Attack:
		is_targeted = config['is_targeted']
		if not is_targeted:
			attack_simulation = AttackSimulation.from_dict(config['attack_simulation'])
		else:
			attack_simulation = None

		return Attack(
			is_targeted=is_targeted,
			data_access=config['data_access'],
			message_access=config['message_access'],
			shadow_model_amount=config['shadow_model_amount'],
			attack_simulation=attack_simulation
		)
