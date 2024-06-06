import numpy as np


def convert_to_csv(path):
	data = np.load(path)
	features = data["features"]
	labels = data["labels"]
	labels = np.argmax(labels, axis=1)
	data = np.column_stack((features, labels))
	new_path = path.replace(".npz", ".csv")
	np.savetxt(new_path, data, delimiter=",")


if __name__ == '__main__':
	# path = ".cache/data/purchase/purchase/purchase100.npz"
	path = ".cache/data/texas/texas/texas100.npz"
	convert_to_csv(path)
