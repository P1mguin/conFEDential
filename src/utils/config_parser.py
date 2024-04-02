import copy
import pickle
from typing import List, Tuple

import yaml

import src.utils as utils


def _adjust_config_values(config, configs: List[dict], *path: Tuple[str]) -> List[dict]:
	"""
	Adjusts a list of configs with the possible values that are described in the config for a given path
	:param config: The config from which the steps should be taken
	:param configs: The configs that will be adjusted
	:param path: The path to the key that describes the steps that will be taken
	"""
	# Get the steps that will be taken
	if config.get("values") is not None:
		steps = config.get("values")
	else:
		step_value = config.get("min")
		maximum = config.get("max")
		step_size = config.get("step_size")
		steps = []
		while step_value <= maximum:
			steps.append(step_value)
			step_value += step_size

	# Adjust the configs
	new_configs = []
	for config in configs:
		for step in steps:
			new_config = copy.deepcopy(config)
			utils.set_dict_value_from_path(new_config, step, *path)
			new_configs.append(new_config)

	return new_configs


def generate_configs_from_batch_config(configs: dict | List[dict], *path: str) -> List[dict]:
	"""
	Loads the possible configs from a batch config
	:param configs: either the batch config, or a helper variable for the recursion that transfers the possible
	configs into several function calls
	:param path: a helper variable for the recursion that transfers which path is being checked
	"""
	if not isinstance(configs, list):
		configs = [configs]

	config = utils.get_dict_value_from_path(configs[0], *path)

	if not isinstance(config, dict):
		return configs

	if config.get("values") is not None or config.get("min") is not None:
		configs = _adjust_config_values(config, configs, *path)
	else:
		for value in config.keys():
			configs = generate_configs_from_batch_config(configs, *path, value)

	return configs


def load_configs_from_batch_config_path(config_path):
	file_path = config_path.replace('.yaml', '.pkl')

	try:
		with open(file_path, "rb") as f:
			return pickle.load(f)
	except FileNotFoundError as _:
		pass

	batch_config = utils.load_yaml_file(config_path)
	configs = utils.generate_configs_from_batch_config(batch_config)

	with open(file_path, "wb") as f:
		pickle.dump(configs, f)

	return configs


def load_yaml_file(yaml_file: str) -> dict:
	"""
	Loads contents of YAML file into a dictionary
	:param yaml_file: absolute path to YAML file
	"""
	with open(yaml_file, 'r') as f:
		return yaml.safe_load(f)
