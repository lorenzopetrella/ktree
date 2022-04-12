import numpy as np

arr = np.load("ktree_history_nobias.npy", allow_pickle=True)
arr.tofile("ktree_history_nobias.csv", sep=";")