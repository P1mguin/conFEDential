import os
import re

import numpy as np
import pandas as pd
from rdp import rdp
from sklearn.metrics import auc
from tqdm import tqdm


# Combine all csvs in results in one big dataframe
def get_roc_curve_df(path):
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


# Average out the ROC curves
def get_average_of_learning_methods(df):
	# Initialize a dictionary to hold the combined data
	combined_data = {'step': df['step']}

	learning_methods = set(re.sub(r'\d+', '', col.split('.')[0]) for col in df.columns if col != 'step')

	for method in learning_methods:
		train_cols = [col for col in df.columns if col.startswith(method) and 'Train' in col]
		val_cols = [col for col in df.columns if col.startswith(method) and 'Validation' in col]

		# Compute the average for the train columns
		if train_cols:
			combined_data[f'{method} - Train'] = df[train_cols].mean(axis=1)

		# Compute the average for the validation columns
		if val_cols:
			combined_data[f'{method} - Validation'] = df[val_cols].mean(axis=1)

		if not train_cols and not val_cols:
			combined_data[method] = df[method]

	# Convert the combined data dictionary to a DataFrame
	combined_df = pd.DataFrame(combined_data)

	return combined_df


# Compute the AUC
# Compute the AUC on the log scale
def store_auc_values(df, dataset, model, locality, attack_type):
	for col in df.columns:
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
		with open("src/results/result.csv", "a") as f:
			f.write(f"{dataset},{model},{locality},{attack_type},{col},{auc_linear},{auc_log}\n")

		# print(f"AUC for {col}:\n\tAUC: {auc_linear}\n\tAUC (log): {auc_log}")


# Downsample the ROC curves to 200 points
def downsample_df(df, target_samples=100):
	downsampled_data = {}

	df = df.sort_values(by="step")

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

			downsampled_data[f"step-{col}"] = downsampled_line[:, 0]
			downsampled_data[col] = downsampled_line[:, 1]
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
	# Create empty result csv file
	with open("src/results/result.csv", "w") as f:
		f.write("dataset,model,locality,attack_type,col,linear,log\n")

	for path, directories, files in os.walk("src/results"):
		if len(files) == 0 or path is "src/results":
			continue

		print(" ".join(path.split("/")[3:]))
		dataset = path.split("/")[3]
		model = path.split("/")[4]
		locality = path.split("/")[5]
		attack_type = translation[path.split("/")[6]]

		# Get all the ROC curves in one dataloader
		roc_curve_df = get_roc_curve_df(path)

		# Compute the AUC of all lines
		# print_auc_values(roc_curve_df)

		# Take the average for multiple learning methods
		roc_curve_df = get_average_of_learning_methods(roc_curve_df)

		# Compute the AUC of combined lines
		store_auc_values(roc_curve_df, dataset, model, locality, attack_type)

		# # Downsample the lines
		# roc_curve_df = downsample_df(roc_curve_df)
		#
		# # Save the file
		# roc_curve_df.to_csv("src/results/result.csv")


if __name__ == '__main__':
	main()
