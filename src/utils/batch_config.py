import copy
from typing import List, Tuple

import yaml

import src.utils as utils
from src.utils.configs import Config


def _adjust_config_values(config: dict, configs: List[dict], *path: Tuple[str]) -> List[dict]:
	"""
	Adjusts a list of configs with the possible values that are described in the config for a given path
	:param config: The config from which the steps should be taken
	:param configs: The configs that will be adjusted
	:param path: The path to the key that describes the steps that will be taken
	"""
	# Get the steps that will be taken
	if config.get("values") is not None:
		steps = config.get("values")
	elif config.get("min") is not None:
		step_value = config.get("min")
		maximum = config.get("max")
		step_size = config.get("step_size")
		steps = []
		while step_value <= maximum:
			steps.append(step_value)
			step_value += step_size
	else:
		raise RuntimeError(
			"Config step value did not find supporting keyword, either try: values, steps, or min-max-step_size")

	# Adjust the configs
	new_configs = []
	for config in configs:
		for step in steps:
			new_config = copy.deepcopy(config)
			utils.set_dict_value_from_path(new_config, step, *path)
			new_configs.append(new_config)

	return new_configs


def _get_stepped_config(config: dict) -> List[dict]:
	"""
	Generates the configurations possible to a stepped experiment
	:param config: The base config
	"""
	paths = utils.find_all_paths(config, "steps")

	step_values = [utils.get_dict_value_from_path(config, *path) for path in paths]
	steps = list(map(list, zip(*step_values)))  # Transpose the stapvalues

	configs = []
	for step in steps:
		new_config = copy.deepcopy(config)
		for i, value in enumerate(step):
			utils.set_dict_value_from_path(new_config, value, *paths[i][:-1])
		configs.append(new_config)

	return configs

def generate_configs_from_yaml_file(file_path: str) -> List[Config]:
	"""
	Generates a list of Config from the path to a batch_configuration YAML file.
	:param file_path: the path to the batch_configuration YAML file
	"""
	with open(file_path, "r") as f:
		yaml_file = yaml.safe_load(f)

	raw_configs = generate_configs_from_batch_config(yaml_file)
	configs = [Config.from_dict(raw_config) for raw_config in raw_configs]
	return configs


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
	elif config.get("steps") is not None:
		return _get_stepped_config(configs[0])
	else:
		for value in config.keys():
			configs = generate_configs_from_batch_config(configs, *path, value)

	return configs
