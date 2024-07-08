import collections

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.stats import entropy
from tqdm import tqdm


def compute_kullback_leibler_divergence(p, q):
	min_value = min(*p, *q)
	p = [x - min_value + 1e-8 for x in p]
	q = [x - min_value + 1e-8 for x in q]

	# Ensure the distributions are valid
	p /= np.sum(p)
	q /= np.sum(q)

	# Ensure the distributions have the same length
	if len(p) != len(q):
		x_p = np.linspace(0, 1, len(p))
		x_q = np.linspace(0, 1, len(q))
		if len(p) < len(q):
			interpolator = interp1d(x_q, q, kind='linear')
			q = interpolator(x_p)
		elif len(p) > len(q):
			interpolator = interp1d(x_p, p, kind='linear')
			p = interpolator(x_q)

		p /= np.sum(p)
		q /= np.sum(q)

	kl_pq = entropy(p, q)
	return kl_pq


def kullback_leibler_analysis(dataloader):
	non_members_dict = {
		"label": [],
		"loss": [],
		"activation": [],
		"gradient": [],
		"metrics": collections.defaultdict(list),
	}
	members_dict = {
		"label": [],
		"loss": [],
		"activation": [],
		"gradient": [],
		"metrics": collections.defaultdict(list),
	}

	for x in tqdm(dataloader.dataset):
		gradient, activation_values, metrics, loss, label = x[0]
		label = torch.argmax(label).item()
		loss = loss.item()
		activation_averages = [np.average(layer) for layer in activation_values]
		gradient_averages = [np.average(layer) for layer in gradient]
		metric_averages = {
			metric_key: [np.average(layer) for layer in metric_values] for metric_key, metric_values in metrics.items()
		}
		if x[1] == 0:
			sample_dict = non_members_dict
		else:
			sample_dict = members_dict
		sample_dict["label"].append(label)
		sample_dict["loss"].append(loss)
		sample_dict["activation"].append(activation_averages)
		sample_dict["gradient"].append(gradient_averages)
		for key, value in metric_averages.items():
			sample_dict["metrics"][key].append(value)

	# Organise the activation, gradient and metric per layer
	members_dict["activation"] = list(map(list, zip(*members_dict["activation"])))
	non_members_dict["activation"] = list(map(list, zip(*non_members_dict["activation"])))
	members_dict["gradient"] = list(map(list, zip(*members_dict["gradient"])))
	non_members_dict["gradient"] = list(map(list, zip(*non_members_dict["gradient"])))
	members_dict["metrics"] = {metrics_key: list(map(list, zip(*metrics_value))) for metrics_key, metrics_value in members_dict["metrics"].items()}
	non_members_dict["metrics"] = {metrics_key: list(map(list, zip(*metrics_value))) for metrics_key, metrics_value in non_members_dict["metrics"].items()}

	# Compute the kullback-leibler divergence for each variable
	loss_kl = compute_kullback_leibler_divergence(non_members_dict["loss"], members_dict["loss"])
	label_kl = compute_kullback_leibler_divergence(non_members_dict["label"], members_dict["label"])
	activation_kls = [compute_kullback_leibler_divergence(non_member, member) for non_member, member in zip(non_members_dict["activation"], members_dict["activation"])]
	gradient_kls = [compute_kullback_leibler_divergence(non_member, member) for non_member, member in zip(non_members_dict["gradient"], members_dict["gradient"])]
	metrics_kls = {
		metric_key: [
			compute_kullback_leibler_divergence(non_member, member) for non_member, member in zip(non_members_dict["metrics"][metric_key], members_dict["metrics"][metric_key])
		]
		for metric_key in members_dict["metrics"].keys()
	}
	return loss_kl, label_kl, activation_kls, gradient_kls, metrics_kls
