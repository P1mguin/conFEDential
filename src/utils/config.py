import yaml


def load_yaml_file(yaml_file):
	with open(yaml_file, 'r') as f:
		return yaml.safe_load(f)


def load_func(function_string):
	namespace = {}
	exec(function_string, namespace)
	return namespace['preprocess_fn']
