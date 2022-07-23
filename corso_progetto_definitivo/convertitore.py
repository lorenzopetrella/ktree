import glob
import numpy as np

files = glob.glob("*.npy")
print(files)

for file in files:
    arr = np.load(file, allow_pickle=True)
    arr.tofile("./fcnn/" + file.split('.')[0] + ".csv", sep=";")
