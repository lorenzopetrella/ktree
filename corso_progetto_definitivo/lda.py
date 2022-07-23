from custom_libraries.miscellaneous import progress_bar
from custom_libraries.image_dataset import *
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from os.path import exists

#############
### SETUP ###
#############

project_folder = ''
weighting = 'paired'
trials = 10
classes = np.load(project_folder + 'results/classes.npy', allow_pickle=True)
save_folder = project_folder + "results/lda/"

##############
### /SETUP ###
##############

for j, (t1, t2, ds) in enumerate(classes):

    filename = save_folder + 'lda_'+ds+'.npy'

    print(f"- Dataset: {ds} / Pair: {t1}-{t2}")

    if exists(filename):
        score = np.load(filename, allow_pickle=True)
    else:
        score = np.zeros(trials)

    test_ds = ImageDataset(ds, 'test', data_dir=None, USPS_dir=project_folder + 'USPS/')
    train_ds = ImageDataset(ds, 'train', data_dir=None, USPS_dir=project_folder + 'USPS/')

    for x in [train_ds, test_ds]:
        x.filter(t1, t2, overwrite=True)
        x.shuffle()
        x.normalize()
        if x.images.shape[1:3] == (28, 28):
            x.pad()
        x.vectorize(True)

    X_train, y_train, _, _ = train_ds.subset(shard=True, shard_number=trials, validation=True,
                                             validation_size=len(test_ds.images))
    average_acc = 0

    for m in range(trials):

        if score[m] != 0:
            continue

        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train[m], y_train[m])
        score[m] = lda.score(test_ds.images, test_ds.labels)
        average_acc = average_acc + score[m]
        print(f"Trial {m+1}: accuracy = {round(score[m]*100, 2)}%")

        # Save accuracy array
        np.save(filename, score)
