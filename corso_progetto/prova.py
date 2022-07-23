import numpy as np


trees_set = [1]

# Load class-dataset list
# classes = np.load(project_folder + 'results/classes.npy', allow_pickle=True)

classes = [[3, 5, 'mnist'],
           [0, 6, 'fmnist'],
           [14, 17, 'emnist'],
           [2, 6, 'kmnist'],
           [3, 5, 'cifar10'],
           [5, 6, 'svhn'],
           [3, 5, 'usps']]


history = np.load('results/fcnn_history.npy', allow_pickle=True)
print("RESULTS:")
for j, (t1, t2, ds) in enumerate(classes):
    print(f"Dataset: {ds} / Pair: {t1}-{t2}")
    for k, trees in enumerate(trees_set):
        print(f"{trees}-tree")
        print(f"Accuracy: {round(np.mean(history[j, :, k, 1]), 4)} ± {round(np.std(history[j, :, k, 1]), 4)}")
        print(history[j, :, k, 1])