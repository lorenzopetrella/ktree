import numpy as np

arr = np.load("ktree_acc_orig_0.npy", allow_pickle=True)
arr.tofile("ktree_acc.csv", sep=";")