project_name = "conFEDential"
from datetime import datetime


def get_wandb_kwargs(config: dict) -> dict:
	"""
	A centralized configuration of what key word args are given to the wandb init command
	:param config: contents of the YAML config
	"""
	time = datetime.now().strftime("%Y-%m-%d_%H:%M")

	return {
		"project": project_name,
		"name": time,
		"config": {
			"dataset": config["dataset"]["name"],
			"model": config["model"]["name"],
			"batch_size": config["simulation"]["batch_size"],
			"client_count": config["simulation"]["client_count"],
			"fraction_fit": config["simulation"]["fraction_fit"],
			"learning_method": config["simulation"]["learning_method"]["optimizer"],
			"local_rounds": config["simulation"]["local_rounds"]
		}
	}
