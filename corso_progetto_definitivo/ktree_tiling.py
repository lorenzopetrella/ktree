import random

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
trees = [2, 4, 8, 16, 32]
classes = [[3, 5, 'mnist']]
save_folder = project_folder + "results/ktree_tile/"
verbose = 0
color_datasets = False

tile_trainables = [True, False]
#tile_trainable = True
tile_epochs = 10

##############
### /SETUP ###
##############

for j, (t1, t2, ds) in enumerate(classes):

    if not color_datasets and ds in ['cifar10', 'svhn']:
        continue

    print(f"Dataset: {ds} / Pair: {t1}-{t2}")

    test_ds = ImageDataset(ds, 'test', data_dir=None)
    train_ds = ImageDataset(ds, 'train', data_dir=None)

    for x in [train_ds, test_ds]:
        x.filter(t1, t2, overwrite=True)
        x.shuffle()
        x.normalize()
        if x.images.shape[1:3] == (28, 28):
            x.pad()
        x.vectorize(True)

    for i in range(trials):

        acc_last_tileTree_filename = save_folder + f"{trees[-1]}tree_rr_not-trainable_{ds}_acc_tileTree.npy"
        if exists(acc_last_tileTree_filename):
            acc_tmp = np.load(acc_last_tileTree_filename, allow_pickle=True)
            if acc_tmp[i] != 0:
                del acc_tmp
                continue
            del acc_tmp

        print(f"Trial {i+1} of {trials}")

        models_layers = []

        test_set = tf.data.Dataset.from_tensor_slices((test_ds.images, test_ds.labels)).batch(bs)

        for k in range(max(trees)):

            print(f"Building tree {k + 1} of {max(trees)}...")

            X_train, y_train, X_valid, y_valid = train_ds.bootstrap(.85, True)
            train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(bs)
            valid_set = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).batch(bs)


            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=epochs),
                         tf.keras.callbacks.ModelCheckpoint(filepath="checkpoints/ktree_checkpoint",
                                                            monitor="val_binary_crossentropy", verbose=0,
                                                            save_best_only=True, save_weights_only=True)]

            model = create_model(input_size=X_train.shape[1], num_trees=1)
            fit_history = model.fit(x=train_set, batch_size=bs, epochs=epochs, validation_data=valid_set,
                                    validation_batch_size=bs, callbacks=callbacks, verbose=verbose)
            print_fit_history(fit_history, epochs)

            model.load_weights("checkpoints/ktree_checkpoint")
            evaluate_history = model.evaluate(x=test_set, batch_size=bs, verbose=0)
            print_evaluate_history(evaluate_history)

            models_layers.append(model.layers)

            del X_train, y_train, X_valid, y_valid, model, callbacks, train_set, valid_set
            gc.collect()

        for tree in trees:

            for tile_trainable in tile_trainables:

                trainable_filename = "trainable" if tile_trainable else "not-trainable"

                acc_tileTree_filename = save_folder + f"{tree}tree_rr_{trainable_filename}_{ds}_acc_tileTree.npy"
                loss_tileTree_filename = save_folder + f"{tree}tree_rr_{trainable_filename}_{ds}_loss_tileTree.npy"

                if exists(acc_tileTree_filename) and exists(loss_tileTree_filename):
                    acc_tileTree = np.load(acc_tileTree_filename, allow_pickle=True)
                    loss_tileTree = np.load(loss_tileTree_filename, allow_pickle=True)
                else:
                    acc_tileTree = np.zeros(trials)
                    loss_tileTree = np.zeros(trials)

                if acc_tileTree[i] != 0:
                    continue

                print(f"Assembling & training {tree}-tree ({'T' if tile_trainable else 'N/T'})...")

                random.shuffle(models_layers)
                layers_weights = []
                for layer_index in range(len(models_layers[0])):
                    layers_weights.append(models_layers[0][layer_index].weights)
                    for model_index in range(1, tree):
                        layers_weights[-1] = tf.concat((layers_weights[-1], models_layers[model_index][layer_index].weights),
                                                       axis=-1)
                    layers_weights[-1] = tf.reshape(layers_weights[-1], shape=[1, -1])

                test_set = tf.data.Dataset.from_tensor_slices((test_ds.images, test_ds.labels)).map(
                    lambda x, y: (tf.tile(x, [tree]), y)).batch(bs)
                X_train, y_train, X_valid, y_valid = train_ds.bootstrap(.85, True)
                train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train)).map(
                    lambda x, y: (tf.tile(x, [tree]), y)).batch(bs)
                valid_set = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).map(
                    lambda x, y: (tf.tile(x, [tree]), y)).batch(bs)

                callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=epochs),
                             tf.keras.callbacks.ModelCheckpoint(filepath="checkpoints/ktree_checkpoint",
                                                                monitor="val_binary_crossentropy", verbose=0,
                                                                save_best_only=True, save_weights_only=True)]

                model = create_model(input_size=X_train.shape[1] * tree, num_trees=tree, trainable=tile_trainable,
                                     weights_initializer=layers_weights)
                fit_history = model.fit(x=train_set, batch_size=bs, epochs=tile_epochs, validation_data=valid_set,
                                        validation_batch_size=bs, callbacks=callbacks, verbose=verbose)
                print_fit_history(fit_history, tile_epochs)

                model.load_weights("checkpoints/ktree_checkpoint")
                model.save_weights(save_folder + f"{tree}tree_rr_{trainable_filename}_{ds}_weights_start_{i+1}")
                evaluate_history = model.evaluate(x=test_set, batch_size=bs, verbose=0)
                print_evaluate_history(evaluate_history)

                (loss_tileTree[i], acc_tileTree[i]) = evaluate_history[1:]
                model.save_weights(save_folder + f"{tree}tree_rr_{trainable_filename}_{ds}_weights-end_{i+1}")
                np.save(acc_tileTree_filename, acc_tileTree, allow_pickle=True)
                np.save(loss_tileTree_filename, loss_tileTree, allow_pickle=True)

                del X_train, y_train, X_valid, y_valid, model, callbacks, train_set, valid_set, test_set
                gc.collect()
