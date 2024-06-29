import collections
import math
import uuid
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def get_auc_curve(roc_auc, fpr, tpr, log_scale: bool = False):
	# Give the plot some id
	plot_id = str(uuid.uuid4())
	_, ax = plt.subplots(num=plot_id, figsize=(5., 5.))

	# Styling
	line_width = 1
	plotted_line_styling = {"lw": line_width, "label": 'ROC curve (area = %0.2f)' % roc_auc}
	diagonal_styling = {"lw": line_width, "label": "Guess (area = 0.50)", "linestyle": "--", "color": "gray"}

	# Plot the data and the guessing line
	ax.plot(fpr, tpr, **plotted_line_styling)
	ax.plot([0.0, 1.0], [0.0, 1.0], **diagonal_styling)

	# Set the axes
	if log_scale:
		ax.set_xscale("log")
		ax.set_yscale("log")
		limits = (1e-5, 1.0)
	else:
		limits = (0.0, 1.0)
	ax.set_xlim(limits)
	ax.set_ylim(limits)

	# Set title, axis-labels, legends, etc.
	ax.set_xlabel('False Positive Rate')
	ax.set_ylabel('True Positive Rate')
	ax.set_title('ROC curve')
	ax.legend(loc="lower right")

	return plt


def _visualize_distributions(
		tensors1: List[Sequence] | Sequence,
		tensors2: List[Sequence] | Sequence,
		labels: List[str],
		titles: List[str] | str,
		is_singular: bool = False,
		log_scale: bool = False,
		save_path: str | None = None
):
	"""
	Visualizes the distributions of two given sequences of distributions. Each distribution is visualized as a KDE plot
	in a subplot where tensors1[i] and tensors2[i] are plotted in the same subplot. The labels are used to label the
	distributions in the legend. The titles are used to label the subplots. The tensors can also represent one distribution
	instead of a sequence, then title must also be singular.
	:param tensors1: A list of sequences of values to plot
	:param tensors2: A list of sequences of values to plot
	:param labels: The labels of the distributions
	:param titles: The titles of the subplots
	:param is_singular: Whether the tensors represent one distribution instead of a sequence
	:param log_scale: Whether to plot the distributions on a log scale
	:param save_path: The path to save the plot to
	"""
	# Expand the tensors if only two distributions (in total) are given
	if is_singular:
		tensors1 = [tensors1]
		tensors2 = [tensors2]
		titles = [titles]

	# Ensure there are equally many distributions
	assert len(tensors1) == len(tensors2)

	# Try to make a square plot for the tensors
	plot_width, plot_height = _get_plot_dimensions(len(tensors1))
	fig_size = (5 * plot_width, 5 * plot_height)

	# Put the tensors in log values if they should be plotted on a log scale
	if log_scale:
		tensors1 = [np.log10(tensor + 1e-8) for tensor in tensors1]
		tensors2 = [np.log10(tensor + 1e-8) for tensor in tensors2]

	# Create a plot with plot_width and plot_height subplots
	fig, axs = plt.subplots(ncols=plot_width, nrows=plot_height, figsize=fig_size, sharex=True, sharey=True)

	for i, (tensor1, tensor2, title) in enumerate(zip(tensors1, tensors2, titles)):
		# Get the correct subplot to plot in
		x = i % plot_width
		y = i // plot_width
		if plot_height > 1:
			ax = axs[y, x]
		elif plot_width > 1:
			ax = axs[x]
		else:
			ax = axs

		# Plot a KDE plot
		sns.kdeplot(tensor1, ax=ax, bw_adjust=0.2, label=labels[0], fill=True, log_scale=log_scale, legend=False)
		sns.kdeplot(tensor2, ax=ax, bw_adjust=0.2, label=labels[1], fill=True, log_scale=log_scale, legend=False)

		# Set the title of the subplot
		ax.set_title(title)
		ax.set_ylabel('')

	# If only one plot is made, the x and y labels are not centred in the plot if we treat it as a collection of subplots
	if is_singular:
		# Set the x and y labels in a larger font
		axs.set_xlabel("Value")
		axs.set_ylabel("Density", x=0.)

		# Place the legend in the top left subplot
		axs.legend(labels=labels, loc="upper left")
	else:
		# Set the x and y labels in a larger font
		fig.supxlabel("Value", fontsize=12)
		fig.supylabel("Density", x=0., fontsize=12)

		# Place the legend in the top left subplot
		fig.legend(labels=labels, loc="upper left", bbox_to_anchor=(0.035, 0.96), fontsize=12)

	# Adjust the layout
	plt.tight_layout()

	# Show the plot
	if not save_path:
		plt.show()
	else:
		plt.savefig(save_path)
		plt.show()
		plt.close()


def visualize_loss_difference(dataloader, log_scale=False, visualize_per_class=False):
	"""
	Visualizes the difference in loss distribution for members and non-members for the entire dataset and per class
	"""
	non_members_dict = collections.defaultdict(list)
	members_dict = collections.defaultdict(list)

	for x in dataloader.dataset:
		label = torch.argmax(x[0][4]).item()
		loss = x[0][3].item()
		if x[1] == 0:
			non_members_dict[label].append(loss)
		else:
			members_dict[label].append(loss)

	non_members_dict = {key: np.array(value) for key, value in non_members_dict.items()}
	members_dict = {key: np.array(value) for key, value in members_dict.items()}

	non_members_loss = np.concatenate(list(non_members_dict.values()))
	members_loss = np.concatenate(list(members_dict.values()))

	_visualize_distributions(
		non_members_loss,
		members_loss,
		["Non-Members", "Members"],
		"Loss distribution over all classes",
		is_singular=True,
		log_scale=log_scale,
		save_path="images/loss_distribution.png"
	)

	if visualize_per_class:
		class_keys = sorted(members_dict.keys())
		members_losses = [np.array([]) if not key in members_dict else members_dict[key] for key in class_keys]
		non_members_losses = [
			np.array([]) if not key in non_members_dict else non_members_dict[key] for key in class_keys
		]
		titles = [f"Loss distribution over class {i}" for i in class_keys]
		_visualize_distributions(
			non_members_losses,
			members_losses,
			["Non-Members", "Members"],
			titles,
			log_scale=log_scale,
			save_path="images/loss_distribution_per_class.png"
		)


def visualize_confidence_difference(dataloader, visualize_per_class=False):
	"""
	Visualizes the difference in confidence in the correct label for members and non-members for the entire dataset and per class
	"""
	non_members_dict = collections.defaultdict(list)
	members_dict = collections.defaultdict(list)

	for x in dataloader.dataset:
		label = torch.argmax(x[0][4]).item()
		confidence = torch.nn.Softmax(dim=0)(x[0][1][-1][0])[label].item()
		if x[1] == 0:
			non_members_dict[label].append(confidence)
		else:
			members_dict[label].append(confidence)

	non_members_dict = {key: np.array(value) for key, value in non_members_dict.items()}
	members_dict = {key: np.array(value) for key, value in members_dict.items()}

	non_members_confidence = np.concatenate(list(non_members_dict.values()))
	members_confidence = np.concatenate(list(members_dict.values()))

	_visualize_distributions(
		non_members_confidence,
		members_confidence,
		["Non-Members", "Members"],
		"Confidence distribution over all classes",
		is_singular=True,
		save_path="images/confidence_distribution.png",
	)

	if visualize_per_class:
		class_keys = sorted(members_dict.keys())
		members_confidences = [np.array([]) if not key in members_dict else members_dict[key] for key in class_keys]
		non_members_confidences = [
			np.array([]) if not key in non_members_dict else non_members_dict[key] for key in class_keys
		]
		_visualize_distributions(
			non_members_confidences,
			members_confidences,
			["Non-Members", "Members"],
			[f"Confidence distribution over class {i}" for i in class_keys],
			save_path="images/confidence_distribution_per_class.png",
		)


def visualize_logit_difference(dataloader, visualize_per_class=False):
	"""
	Visualizes the difference in logit corrected confidence in the correct label for members and non-members for the entire dataset and per class
	"""
	non_members_dict = collections.defaultdict(list)
	members_dict = collections.defaultdict(list)

	for x in dataloader.dataset:
		label = torch.argmax(x[0][4]).item()
		confidence = x[0][1][-1][0][label].item()
		if x[1] == 0:
			non_members_dict[label].append(confidence)
		else:
			members_dict[label].append(confidence)

	non_members_dict = {key: np.array(value) for key, value in non_members_dict.items()}
	members_dict = {key: np.array(value) for key, value in members_dict.items()}

	non_members_confidence = np.concatenate(list(non_members_dict.values()))
	members_confidence = np.concatenate(list(members_dict.values()))

	_visualize_distributions(
		non_members_confidence,
		members_confidence,
		["Non-Members", "Members"],
		"Logit confidence distribution over all classes",
		is_singular=True,
		save_path="images/logit_confidence_distribution.png",
	)

	if visualize_per_class:
		class_keys = sorted(members_dict.keys())
		members_confidences = [np.array([]) if not key in members_dict else members_dict[key] for key in class_keys]
		non_members_confidences = [
			np.array([]) if not key in non_members_dict else non_members_dict[key] for key in class_keys
		]
		_visualize_distributions(
			non_members_confidences,
			members_confidences,
			["Non-Members", "Members"],
			[f"Logit confidence distribution over class {i}" for i in class_keys],
			save_path="images/logit_confidence_distribution_per_class.png",
		)


def _get_plot_dimensions(n_plots):
	"""
	Helper function to get the dimensions of the plot
	"""
	ncols = int(math.sqrt(n_plots))
	nrows = math.ceil(n_plots / ncols)

	# Adjust if ncols and nrows don't multiply to n exactly
	while nrows * ncols != n_plots:
		ncols += 1
		nrows = math.ceil(n_plots / ncols)

	return ncols, nrows
