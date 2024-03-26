import yaml


def load_yaml_file(yaml_file: str) -> dict:
	"""
	Loads contents of YAML file into a dictionary
	:param yaml_file: absolute path to YAML file
	"""
	with open(yaml_file, 'r') as f:
		return yaml.safe_load(f)
