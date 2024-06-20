import collections
import uuid

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


def _visualize_distribution(tensor1, tensor2, labels, title, log_scale=False):
	if log_scale:
		tensor1 = np.log10(tensor1)
		tensor2 = np.log10(tensor2)

	plt.figure()

	sns.kdeplot(tensor1, bw_adjust=0.2, label=labels[0], fill=True, log_scale=log_scale)
	sns.kdeplot(tensor2, bw_adjust=0.2, label=labels[1], fill=True, log_scale=log_scale)

	plt.title(title)
	plt.xlabel("Value")
	plt.ylabel("Frequency")

	# Place the legend top left
	plt.legend(loc="upper left")
	plt.show()


def visualize_loss_difference(dataloader, visualize_per_class=False):
	"""
	Visualizes the difference in loss distribution for members and non-members for the entire dataset and per class
	"""
	non_members_dict = collections.defaultdict(list)
	members_dict = collections.defaultdict(list)

	for x in dataloader.dataset:
		label = np.argmax(x[0][4]).item()
		loss = x[0][3].item()
		if x[1] == 0:
			non_members_dict[label].append(loss)
		else:
			members_dict[label].append(loss)

	non_members_dict = {key: np.array(value) for key, value in non_members_dict.items()}
	members_dict = {key: np.array(value) for key, value in members_dict.items()}

	non_members_loss = np.concatenate(list(non_members_dict.values()))
	members_loss = np.concatenate(list(members_dict.values()))

	_visualize_distribution(
		non_members_loss,
		members_loss,
		["Non-Members", "Members"],
		"(Test) Loss distribution over all classes",
		log_scale=True
	)

	if visualize_per_class:
		for i, (non_members_loss, members_loss) in enumerate(zip(non_members_dict.values(), members_dict.values())):
			_visualize_distribution(
				non_members_loss,
				members_loss,
				["Non-Members", "Members"],
				f"(Test) Loss distribution over class {i}",
				log_scale=True
			)

def visualize_confidence_difference(dataloader, visualize_per_class=False):
	"""
	Visualizes the difference in confidence in the correct label for members and non-members for the entire dataset and per class
	"""
	non_members_dict = collections.defaultdict(list)
	members_dict = collections.defaultdict(list)

	for x in dataloader.dataset:
		label = np.argmax(x[0][4]).item()
		confidence = torch.nn.Softmax(dim=0)(x[0][1][-1][0])[label].item()
		if x[1] == 0:
			non_members_dict[label].append(confidence)
		else:
			members_dict[label].append(confidence)

	non_members_dict = {key: np.array(value) for key, value in non_members_dict.items()}
	members_dict = {key: np.array(value) for key, value in members_dict.items()}

	non_members_confidence = np.concatenate(list(non_members_dict.values()))
	members_confidence = np.concatenate(list(members_dict.values()))

	_visualize_distribution(non_members_confidence, members_confidence, ["Non-Members", "Members"], "(Test) Confidence distribution over all classes")

	if visualize_per_class:
		for i, (non_members_confidence, members_confidence) in enumerate(zip(non_members_dict.values(), members_dict.values())):
			_visualize_distribution(
				non_members_confidence,
				members_confidence,
				["Non-Members", "Members"],
				f"(Test) Confidence distribution over class {i}"
			)


def visualize_logit_difference(dataloader, visualize_per_class=False):
	"""
	Visualizes the difference in logit corrected confidence in the correct label for members and non-members for the entire dataset and per class
	"""
	non_members_dict = collections.defaultdict(list)
	members_dict = collections.defaultdict(list)

	for x in dataloader.dataset:
		label = np.argmax(x[0][4]).item()
		confidence = x[0][1][-1][0][label].item()
		if x[1] == 0:
			non_members_dict[label].append(confidence)
		else:
			members_dict[label].append(confidence)

	non_members_dict = {key: np.array(value) for key, value in non_members_dict.items()}
	members_dict = {key: np.array(value) for key, value in members_dict.items()}

	non_members_confidence = np.concatenate(list(non_members_dict.values()))
	members_confidence = np.concatenate(list(members_dict.values()))

	_visualize_distribution(non_members_confidence, members_confidence, ["Non-Members", "Members"], "(Test) Logit confidence distribution over all classes")

	if visualize_per_class:
		for i, (non_members_confidence, members_confidence) in enumerate(zip(non_members_dict.values(), members_dict.values())):
			_visualize_distribution(
				non_members_confidence,
				members_confidence,
				["Non-Members", "Members"],
				f"(Test) Logit confidence distribution over class {i}"
			)
