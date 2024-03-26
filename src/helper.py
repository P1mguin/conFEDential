from collections import OrderedDict

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_weights(model):
	return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model, weights) -> None:
	params_dict = zip(model.state_dict().keys(), weights)
	state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
	model.load_state_dict(state_dict, strict=True)


def train(epochs, parameters, Net, trainloader, Criterion, Optimizer):
	"""Train the network on the training set."""
	net = Net().to(DEVICE)

	if parameters is not None:
		set_weights(net, parameters)

	criterion = Criterion()
	optimizer = Optimizer(net.parameters())
	for _ in range(epochs):
		for features, labels in trainloader:
			features, labels = features.to(DEVICE), labels.to(DEVICE)
			optimizer.zero_grad()
			loss = criterion(net(features), labels)
			loss.backward()
			optimizer.step()

	# Prepare return values
	parameters = get_weights(net)
	data_size = len(trainloader)
	return parameters, data_size


def test(parameters, Net, testloader, Criterion):
	"""Validate the network on the entire test set."""
	# Create model
	net = Net().to(DEVICE)

	# Load weights
	if parameters is not None:
		set_weights(net, parameters)

	criterion = Criterion()
	correct, total, loss = 0, 0, 0.0
	with torch.no_grad():
		for data in testloader:
			features, labels = data['x'].to(DEVICE), data['y'].to(DEVICE)
			outputs = net(features)
			loss += criterion(outputs, labels).item()
			if outputs.data.shape[1] == 1:
				predicted = torch.round(outputs.data)
			else:
				predicted = torch.max(outputs.data, 1)[1]
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	accuracy = correct / total

	data_size = len(testloader)
	return loss, accuracy, data_size
