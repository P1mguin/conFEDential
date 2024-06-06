import uuid

import matplotlib.pyplot as plt


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
