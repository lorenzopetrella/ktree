from custom_libraries.miscellaneous import *
from custom_libraries.image_dataset import *
from custom_libraries.ktree import *
import numpy as np
from os.path import exists
import gc

#############
### SETUP ###
#############

project_folder = ''
bs = 256
trials = 10
epochs = 2000
trees_set = [1]
classes = np.load(project_folder + 'results/classes.npy', allow_pickle=True)
save_folder = project_folder + "results/ktree/"
verbose = 0
bias = False
non_negative = False
color_datasets = False

##############
### /SETUP ###
##############

if bias:
    bias_filename = 'bias'
else:
    bias_filename = 'nobias'

if non_negative:
    non_negative_filename = 'nonneg'
else:
    non_negative_filename = 'real'

for j, (t1, t2, ds) in enumerate(classes):

    if not color_datasets and ds in ['cifar10', 'svhn']:
        continue

    print(f"Dataset: {ds} / Pair: {t1}-{t2}")

    test_ds = ImageDataset(ds, 'test', data_dir=None, shuffle_files=False)
    train_ds = ImageDataset(ds, 'train', data_dir=None, shuffle_files=False)
    test_ds_2 = ImageDataset(ds, 'test', data_dir=None, shuffle_files=False)
    train_ds_2 = ImageDataset(ds, 'train', data_dir=None, shuffle_files=False)

    for x in [train_ds, test_ds, train_ds_2, test_ds_2]:
        x.filter(t1, t2, overwrite=True)
        x.normalize()
        if x.images.shape[1:3] == (28, 28):
            x.pad()

    for x in [train_ds, test_ds]:
        x.vectorize(merge_channels=True, by_row=True)

    for x in [train_ds_2, test_ds_2]:
        x.vectorize(merge_channels=True, by_row=False)

    for (x, y) in [(train_ds, train_ds_2), (test_ds, test_ds_2)]:
        x.images = np.concatenate((x.images, y.images), axis=1)
        x.labels = np.concatenate((x.labels, y.labels), axis=None)
        x.shuffle()

    del train_ds_2, test_ds_2

    for k, trees in enumerate(trees_set):

        filename_acc = save_folder + f"{trees}tree_rowcol_{bias_filename}_{non_negative_filename}_{ds}_acc.npy"
        filename_loss = save_folder + f"{trees}tree_rowcol_{bias_filename}_{non_negative_filename}_{ds}_loss.npy"
        if exists(filename_acc) and exists(filename_loss):
            acc = np.load(filename_acc, allow_pickle=True)
            loss = np.load(filename_loss, allow_pickle=True)
        else:
            acc = np.zeros(trials)
            loss = np.zeros(trials)

        print(f"{trees}-tree")

        test_set = tf.data.Dataset.from_tensor_slices((test_ds.images, test_ds.labels)).map(
            lambda x, y: (tf.tile(x, [trees]), y)).batch(bs)

        for i in range(trials):

            if acc[i] != 0:
                continue

            print(f"Trial {i + 1}")

            X_train, y_train, X_valid, y_valid = train_ds.bootstrap(.85, True)
            train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train)).map(
                lambda x, y: (tf.tile(x, [trees]), y)).batch(bs)
            valid_set = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).map(
                lambda x, y: (tf.tile(x, [trees]), y)).batch(bs)

            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=epochs),
                         tf.keras.callbacks.ModelCheckpoint(filepath="checkpoints/ktree_checkpoint",
                                                            monitor="val_binary_crossentropy", verbose=0,
                                                            save_best_only=True, save_weights_only=True)]

            model = create_model(input_size=X_train.shape[1] * trees, num_trees=trees*2, use_bias=bias,
                                 non_neg=non_negative)
            fit_history = model.fit(x=train_set, batch_size=bs, epochs=epochs, validation_data=valid_set,
                                    validation_batch_size=bs, callbacks=callbacks, verbose=verbose)
            print_fit_history(fit_history, epochs)

            model.load_weights("checkpoints/ktree_checkpoint")
            evaluate_history = model.evaluate(x=test_set, batch_size=bs, verbose=0)
            print_evaluate_history(evaluate_history)

            (loss[i], acc[i]) = evaluate_history[1:]

            np.save(filename_acc, acc, allow_pickle=True)
            np.save(filename_loss, loss, allow_pickle=True)

            del model, train_set, valid_set, X_train, y_train, X_valid, y_valid
            gc.collect()
