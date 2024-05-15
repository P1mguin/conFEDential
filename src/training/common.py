from collections import OrderedDict
from typing import List

import numpy.typing as npt
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_weights(model: nn.Module) -> List[npt.NDArray]:
	"""
	Takes a PyTorch model and returns its parameters as a list of NumPy arrays
	:param model: the PyTorch model
	"""
	return [val.cpu().numpy().copy() for _, val in model.state_dict().items()]


def set_weights(model: nn.Module, weights: List[npt.NDArray]) -> None:
	"""
	Takes a PyTorch model and a list of NumPy arrays and makes the NumPy arrays the weights of the model
	:param model: the PyTorch model
	:param weights: the target weights
	"""
	params_dict = zip(model.state_dict().keys(), weights)
	state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
	model.load_state_dict(state_dict, strict=True)
