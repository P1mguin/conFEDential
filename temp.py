from src.experiment import Config


def main():
	yaml_file = "examples/targeted/mnist/logistic_regression/fed_nag.yaml"
	config = Config.from_yaml_file(yaml_file)

	client_resources = {
		"num_cpus": 4,
		"num_gpus": 0
	}
	is_capturing = True
	is_online = False
	run_name = None

	simulation = config.simulation
	simulation._simulate_federation(client_resources, is_capturing, is_online, run_name)


if __name__ == '__main__':
	main()
