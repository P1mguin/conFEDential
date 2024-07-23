import collections
from logging import INFO

import numpy as np
import torch
from flwr.common import log
from scipy.interpolate import interp1d
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, wasserstein_distance
from tqdm import tqdm


def _sample_vectors_to_same_size(p, q, log_scale=False):
	if log_scale:
		p = np.log10(p + 1e-8)
		q = np.log10(q + 1e-8)
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

def compute_wasserstein_distance(p, q, log_scale=False):
	p, q = _sample_vectors_to_same_size(p, q, log_scale)
	wasserstein = wasserstein_distance(p, q)
	return wasserstein

def compute_mean_difference(p, q):
	p_mean = np.mean(p)
	q_mean = np.mean(q)
	return p_mean - q_mean

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

	# Compute the wasserstein distance for each variable
	loss_wass = compute_wasserstein_distance(non_members_dict["loss"], members_dict["loss"])
	label_wass = compute_wasserstein_distance(non_members_dict["label"], members_dict["label"])
	activation_wasss = [compute_wasserstein_distance(non_member, member) for non_member, member in zip(non_members_dict["activation"], members_dict["activation"])]
	gradient_wasss = [compute_wasserstein_distance(non_member, member) for non_member, member in zip(non_members_dict["gradient"], members_dict["gradient"])]
	metrics_wasss = {
		metric_key: [
			compute_wasserstein_distance(non_member, member) for non_member, member in zip(non_members_dict["metrics"][metric_key], members_dict["metrics"][metric_key])
		]
		for metric_key in members_dict["metrics"].keys()
	}
	log(INFO, f"Loss Wasserstein: {loss_wass}")
	log(INFO, f"Label Wasserstein: {label_wass}")
	log(INFO, f"Activation Wasserstein: {activation_wasss}")
	log(INFO, f"Gradient Wasserstein: {gradient_wasss}")
	log(INFO, f"Metrics Wasserstein: {metrics_wasss}")

	# Compute the wasserstein distance along the log distribution for each variable
	loss_wass = compute_wasserstein_distance(non_members_dict["loss"], members_dict["loss"], log_scale=True)
	label_wass = compute_wasserstein_distance(non_members_dict["label"], members_dict["label"], log_scale=True)
	activation_wasss = [compute_wasserstein_distance(non_member, member, log_scale=True) for non_member, member in zip(non_members_dict["activation"], members_dict["activation"])]
	gradient_wasss = [compute_wasserstein_distance(non_member, member, log_scale=True) for non_member, member in zip(non_members_dict["gradient"], members_dict["gradient"])]
	metrics_wasss = {
		metric_key: [
			compute_wasserstein_distance(non_member, member, log_scale=True) for non_member, member in zip(non_members_dict["metrics"][metric_key], members_dict["metrics"][metric_key])
		]
		for metric_key in members_dict["metrics"].keys()
	}
	log(INFO, f"Loss Log Wasserstein: {loss_wass}")
	log(INFO, f"Label Log Wasserstein: {label_wass}")
	log(INFO, f"Activation Log Wasserstein: {activation_wasss}")
	log(INFO, f"Gradient Log Wasserstein: {gradient_wasss}")
	log(INFO, f"Metrics Log Wasserstein: {metrics_wasss}")

	# Compute the difference between the means for each variable
	loss_mean = compute_mean_difference(non_members_dict["loss"], members_dict["loss"])
	label_mean = compute_mean_difference(non_members_dict["label"], members_dict["label"])
	activation_means = [compute_mean_difference(non_member, member) for non_member, member in zip(non_members_dict["activation"], members_dict["activation"])]
	gradient_means = [compute_mean_difference(non_member, member) for non_member, member in zip(non_members_dict["gradient"], members_dict["gradient"])]
	metrics_means = {
		metric_key: [
			compute_mean_difference(non_member, member) for non_member, member in zip(non_members_dict["metrics"][metric_key], members_dict["metrics"][metric_key])
		]
		for metric_key in members_dict["metrics"].keys()
	}
	log(INFO, f"Loss Mean Difference: {loss_mean}")
	log(INFO, f"Label Mean Difference: {label_mean}")
	log(INFO, f"Activation Mean Difference: {activation_means}")
	log(INFO, f"Gradient Mean Difference: {gradient_means}")
	log(INFO, f"Metrics Mean Difference: {metrics_means}")


def normalize(data):
	return (data - data.min()) / (data.max() - data.min())
