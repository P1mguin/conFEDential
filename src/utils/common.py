from typing import Any, Tuple


def get_dict_value_from_path(dictionary: dict, *path: Tuple[str]) -> Any:
	"""
	Common function that gets a value from a dict given a path
	:param dictionary: the dictionary to get value from
	:param path: the path of the value
	"""
	value = dictionary
	for key in path:
		value = value[key]
	return value


def set_dict_value_from_path(dictionary: dict, new_value: Any, *path: Tuple[str]) -> dict:
	"""
	Common function that sets a value from a dict given a path, sets it in place but also returns the updated dictionary
	:param dictionary: the dictionary on which the value should be set
	:param new_value: the new value to set
	:param path: the path at which the new value should be set
	:return:
	"""
	for key in path[:-1]:
		dictionary = dictionary[key]

	dictionary[path[-1]] = new_value
	return dictionary
