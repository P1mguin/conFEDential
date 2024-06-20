import collections
import uuid

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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


def _visualize_distribution(tensor1, tensor2, labels, title, log_scale=True):
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


def visualize_loss_difference(dataloader):
	"""
	Visualizes the difference in loss distribution for members and non-members for the entire dataset and per class
	"""
	non_members_dict = collections.defaultdict(list)
	members_dict = collections.defaultdict(list)

	for x in dataloader.dataset:
		if x[1] == 0:
			non_members_dict[np.argmax(x[0][4]).item()].append(x[0][3].item())
		else:
			members_dict[np.argmax(x[0][4]).item()].append(x[0][3].item())

	non_members_dict = {key: np.array(value) for key, value in non_members_dict.items()}
	members_dict = {key: np.array(value) for key, value in members_dict.items()}

	non_members_loss = np.concatenate(list(non_members_dict.values()))
	members_loss = np.concatenate(list(members_dict.values()))

	_visualize_distribution(non_members_loss, members_loss, ["Non-Members", "Members"],
							"(Test) Loss distribution over all classes")

	for i, (non_members_loss, members_loss) in enumerate(zip(non_members_dict.values(), members_dict.values())):
		_visualize_distribution(
			non_members_loss,
			members_loss,
			["Non-Members", "Members"],
			f"(Test) Loss distribution over class {i}"
		)
