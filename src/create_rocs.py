import os
import re

import numpy as np
import pandas as pd
from rdp import rdp
from sklearn.metrics import auc
from tqdm import tqdm


def get_roc_curve_df(path: str):
	"""
	Combine all the ROC curves in the results folder into one dataframe
	:param path: the path in which the ROC curves are described
	"""
	# For all files in the results folder, take the 'linekey', the 'lineval' and the 'step' column as a dataframe
	dataframes = []
	guessing_appended = False
	for file_name in os.listdir(path):
		if not file_name.endswith("csv"):
			continue
		df = pd.read_csv(f"{path}/{file_name}", usecols=["lineKey", "lineVal", "step"])
		df["lineKey"] = df["lineKey"].str.extract(r'(\w+)')
		df = df.pivot_table(index="step", columns="lineKey", values="lineVal")
		df.rename(columns={"Train": f"{file_name} - Train", "Validation": f"{file_name} - Validation"}, inplace=True)
		df.reset_index(inplace=True)
		if not guessing_appended:
			dataframes.append(df)
			guessing_appended = True
		else:
			dataframes.append(df[["step", f"{file_name} - Train", f"{file_name} - Validation"]])

	# Combine the dataframes into one dataframe where step is the value to merge on
	final_df = dataframes[0]
	for df in dataframes[1:]:
		final_df = pd.merge(final_df, df, on="step", how="outer")
	return final_df.sort_values(by="step")


def get_average_of_learning_methods(df: pd.DataFrame):
	"""
	Combine multiple results of the same learning method into one
	:param df: the dataframe in which the results are stored
	"""
	# Initialize a dictionary to hold the combined data
	combined_data = {'step': df['step']}

	# Get all the learning methods
	learning_methods = set(re.sub(r'\d+', '', col.split('.')[0]) for col in df.columns if col != 'step')

	for method in learning_methods:
		# Get the columns for the current method
		train_cols = [col for col in df.columns if col.startswith(method) and 'Train' in col]
		val_cols = [col for col in df.columns if col.startswith(method) and 'Validation' in col]

		# Compute the average for the train columns
		if train_cols:
			combined_data[f'{method} - Train'] = df[train_cols].mean(axis=1)

		# Compute the average for the validation columns
		if val_cols:
			combined_data[f'{method} - Validation'] = df[val_cols].mean(axis=1)

		# If there are no train or validation columns, just copy the data
		if not train_cols and not val_cols:
			combined_data[method] = df[method]

	# Convert the combined data dictionary to a DataFrame
	combined_df = pd.DataFrame(combined_data)
	return combined_df


def print_auc_values(df: pd.DataFrame):
	"""
	Print the AUC values for the ROC curves
	:param df: the dataframe with the ROC curves
	"""
	# For all columns in the dataframe, compute the AUC for linear and logarithmic axes
	for col in df.columns:
		# Do not compute the AUC for the x-axis
		if col == "step":
			continue

		# Drop NAN values for x and y
		valid_data = df[['step', col]].dropna()
		x = valid_data['step']
		y = valid_data[col]

		# Compute AUC for linear axes
		auc_linear = auc(x, y)

		# Compute AUC for logarithmic axes
		positive_data = valid_data[(valid_data['step'] > 0) & (valid_data[col] > 0)]
		log_x = np.log(positive_data['step'])
		log_y = np.log(positive_data[col])
		if len(positive_data) == 1:
			log_x = log_x.append(pd.Series([-1.0]))
			log_y = log_y.append(pd.Series([-1.0]))
		log_x = -(log_x - min(log_x)) / min(log_x) + 1e-9
		log_y = -(log_y - min(log_y)) / min(log_y) + 1e-9
		auc_log = auc(log_x, log_y)
		print(f"AUC for {col}:\n\tAUC: {auc_linear}\n\tAUC (log): {auc_log}")


def downsample_df(df: pd.DataFrame, target_samples: int = 100):
	"""
	Downsample the dataframe to have a target number of samples
	:param df: the dataframe which columns to downsample
	:param target_samples: the target number of samples
	"""
	# Initialize a dictionary to hold the downsampled data
	downsampled_data = {}

	# Sort the dataframe by step
	df = df.sort_values(by="step")

	# For all columns in the dataframe, downsample the data
	for col in tqdm(df.columns):
		if col == 'step':
			continue
		else:
			line = df[["step", col]].to_numpy()

			# Define the range for epsilon
			epsilon_min = 0.0
			epsilon_max = 1.0

			# Perform binary search to find the optimal epsilon
			for _ in range(50):  # Limit the number of iterations to prevent infinite loop
				epsilon = (epsilon_min + epsilon_max) / 2.0
				downsampled_line = np.array(rdp(line, epsilon=epsilon))
				if len(downsampled_line) > target_samples:
					epsilon_min = epsilon
				elif len(downsampled_line) < target_samples:
					epsilon_max = epsilon
				else:
					break  # Stop if we've found an epsilon that gives us the target number of samples

			# Store the downsampled data in the dictionary
			downsampled_data[f"step-{col}"] = downsampled_line[:, 0]
			downsampled_data[col] = downsampled_line[:, 1]

	# Convert the downsampled data dictionary to a DataFrame
	downsampled_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in downsampled_data.items()]))

	# Add an infinitesimally small value to each column so the log plot closes
	for col in downsampled_df.columns:
		downsampled_df[col] += 1e-9
	return downsampled_df

translation = {
	"100": "A",
	"110": "B",
	"101": "C",
	"111": "D"
}
def main():
	for path, directories, files in os.walk("src/results"):
		if len(files) == 0 or path == "src/results":
			continue

		# Print the path
		print(" ".join(path.split("/")[2:]))

		# Get all the ROC curves in one dataloader
		roc_curve_df = get_roc_curve_df(path)

		# Take the average for multiple learning methods
		roc_curve_df = get_average_of_learning_methods(roc_curve_df)

		# Compute the AUC of combined lines
		print_auc_values(roc_curve_df)

		# Downsample the lines
		roc_curve_df = downsample_df(roc_curve_df)

		# Save the file
		roc_curve_df.to_csv(f"{path}/result.csv")


if __name__ == '__main__':
	main()
