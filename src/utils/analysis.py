import collections
from logging import INFO

import numpy as np
import torch
from flwr.common import log
from scipy.interpolate import interp1d
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from tqdm import tqdm


def _sample_vectors_to_same_size(p, q):
	# Ensure the distributions are normalized
	p = normalize(p) + 1e-8
	q = normalize(q) + 1e-8

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

	p = normalize(p) + 1e-8
	q = normalize(q) + 1e-8
	return p, q


def compute_jensen_shannon_divergence(p, q):
	p, q = _sample_vectors_to_same_size(p, q)
	js = jensenshannon(p, q)
	return js


def compute_kullback_leibler_divergence(p, q):
	p, q = _sample_vectors_to_same_size(p, q)
	kl_pq = entropy(p, q)
	return kl_pq


def do_analyses(dataloader):
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
	members_dict["activation"] = np.array(list(map(list, zip(*members_dict["activation"]))))
	non_members_dict["activation"] = np.array(list(map(list, zip(*non_members_dict["activation"]))))
	members_dict["gradient"] = np.array(list(map(list, zip(*members_dict["gradient"]))))
	non_members_dict["gradient"] = np.array(list(map(list, zip(*non_members_dict["gradient"]))))
	members_dict["metrics"] = {metrics_key: np.array(list(map(list, zip(*metrics_value)))) for
							   metrics_key, metrics_value in members_dict["metrics"].items()}
	non_members_dict["metrics"] = {metrics_key: np.array(list(map(list, zip(*metrics_value)))) for
								   metrics_key, metrics_value in non_members_dict["metrics"].items()}

	# Convert all other variables to NumPy array
	members_dict["label"] = np.array(members_dict["label"])
	non_members_dict["label"] = np.array(non_members_dict["label"])
	members_dict["loss"] = np.array(members_dict["loss"])
	non_members_dict["loss"] = np.array(non_members_dict["loss"])

	# Compute the Jensen-Shannon divergence for each variable
	loss_js = compute_jensen_shannon_divergence(non_members_dict["loss"], members_dict["loss"])
	label_js = compute_jensen_shannon_divergence(non_members_dict["label"], members_dict["label"])
	activation_jss = [compute_jensen_shannon_divergence(non_member, member) for non_member, member in
					  zip(non_members_dict["activation"], members_dict["activation"])]
	gradient_jss = [compute_jensen_shannon_divergence(non_member, member) for non_member, member in
					zip(non_members_dict["gradient"], members_dict["gradient"])]
	metrics_jss = {
		metric_key: [
			compute_jensen_shannon_divergence(non_member, member) for non_member, member in
			zip(non_members_dict["metrics"][metric_key], members_dict["metrics"][metric_key])
		]
		for metric_key in members_dict["metrics"].keys()
	}
	log(INFO, f"Loss JS: {loss_js}")
	log(INFO, f"Label JS: {label_js}")
	log(INFO, f"Activation JS: {activation_jss}")
	log(INFO, f"Gradient JS: {gradient_jss}")
	log(INFO, f"Metrics JS: {metrics_jss}")

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
	log(INFO, f"Loss KL: {loss_kl}")
	log(INFO, f"Label KL: {label_kl}")
	log(INFO, f"Activation KL: {activation_kls}")
	log(INFO, f"Gradient KL: {gradient_kls}")
	log(INFO, f"Metrics KL: {metrics_kls}")


def normalize(data):
	return (data - data.min()) / (data.max() - data.min())
