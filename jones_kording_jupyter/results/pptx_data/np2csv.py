import numpy as np

arr = np.load("ktree_acc_orig_usps.npy", allow_pickle=True)
arr.tofile("ktree_acc_usps_bias.csv", sep=";")