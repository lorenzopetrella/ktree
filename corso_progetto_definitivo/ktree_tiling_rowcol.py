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
trees = 2
classes = np.load(project_folder + 'results/classes.npy', allow_pickle=True)
save_folder = project_folder + "results/ktree_tile/"
verbose = 0
color_datasets = False

tile_trainable = False
tile_epochs = 10

##############
### /SETUP ###
##############

if tile_trainable:
    trainable_filename = "trainable"
else:
    trainable_filename = "not-trainable"

for j, (t1, t2, ds) in enumerate(classes):

    acc_firstTree_filename = save_folder + f"{trees}tree_rowcol_{trainable_filename}_{ds}_acc_firstTree.npy"
    acc_secondTree_filename = save_folder + f"{trees}tree_rowcol_{trainable_filename}_{ds}_acc_secondTree.npy"
    acc_tileTree_filename = save_folder + f"{trees}tree_rowcol_{trainable_filename}_{ds}_acc_tileTree.npy"
    loss_firstTree_filename = save_folder + f"{trees}tree_rowcol_{trainable_filename}_{ds}_loss_firstTree.npy"
    loss_secondTree_filename = save_folder + f"{trees}tree_rowcol_{trainable_filename}_{ds}_loss_secondTree.npy"
    loss_tileTree_filename = save_folder + f"{trees}tree_rowcol_{trainable_filename}_{ds}_loss_tileTree.npy"

    if exists(acc_firstTree_filename) and exists(acc_secondTree_filename) and exists(acc_tileTree_filename) and exists(
            loss_firstTree_filename) and exists(loss_secondTree_filename) and exists(loss_tileTree_filename):
        acc_firstTree = np.load(acc_firstTree_filename, allow_pickle=True)
        acc_secondTree = np.load(acc_secondTree_filename, allow_pickle=True)
        acc_tileTree = np.load(acc_tileTree_filename, allow_pickle=True)
        loss_firstTree = np.load(loss_firstTree_filename, allow_pickle=True)
        loss_secondTree = np.load(loss_secondTree_filename, allow_pickle=True)
        loss_tileTree = np.load(loss_tileTree_filename, allow_pickle=True)
    else:
        acc_firstTree = np.zeros(trials)
        acc_secondTree = np.zeros(trials)
        acc_tileTree = np.zeros(trials)
        loss_firstTree = np.zeros(trials)
        loss_secondTree = np.zeros(trials)
        loss_tileTree = np.zeros(trials)

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

        print(f"Trial {i + 1}")

        if acc_tileTree[i] != 0:
            continue

        models_layers = []

        for k in range(trees):

            print(f"Tree #{k + 1}")

            test_ds = ImageDataset(ds, 'test', data_dir=None)
            train_ds = ImageDataset(ds, 'train', data_dir=None)

            for x in [train_ds, test_ds]:
                x.filter(t1, t2, overwrite=True)
                x.shuffle()
                x.normalize()
                if x.images.shape[1:3] == (28, 28):
                    x.pad()
                x.vectorize(True, by_row=(k % 2 == 0))

            X_train, y_train, X_valid, y_valid = train_ds.bootstrap(.85, True)
            train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(bs)
            valid_set = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).batch(bs)
            test_set = tf.data.Dataset.from_tensor_slices((test_ds.images, test_ds.labels)).batch(bs)

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

            if k == 0:
                (loss_firstTree[i], acc_firstTree[i]) = evaluate_history[1:]
                model.save_weights(save_folder + f"first_rc_{trainable_filename}_t{i + 1}_weights")
                np.save(acc_firstTree_filename, acc_firstTree, allow_pickle=True)
                np.save(loss_firstTree_filename, loss_firstTree, allow_pickle=True)
            else:
                (loss_secondTree[i], acc_secondTree[i]) = evaluate_history[1:]
                model.save_weights(save_folder + f"second_rc_{trainable_filename}_t{i + 1}_weights")
                np.save(acc_secondTree_filename, acc_secondTree, allow_pickle=True)
                np.save(loss_secondTree_filename, loss_secondTree, allow_pickle=True)

            models_layers.append(model.layers)

        print(f"Tile Tree")

        layers_weights = []
        for layer_index in range(len(models_layers[0])):
            layers_weights.append(models_layers[0][layer_index].weights)
            for model_index in range(1, len(models_layers)):
                layers_weights[-1] = tf.concat((layers_weights[-1], models_layers[model_index][layer_index].weights),
                                               axis=-1)
            layers_weights[-1] = tf.reshape(layers_weights[-1], shape=[1, -1])

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

        test_set = tf.data.Dataset.from_tensor_slices((test_ds.images, test_ds.labels)).batch(bs)
        X_train, y_train, X_valid, y_valid = train_ds.bootstrap(.85, True)
        train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(bs)
        valid_set = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).batch(bs)

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=epochs),
                     tf.keras.callbacks.ModelCheckpoint(filepath="checkpoints/ktree_checkpoint",
                                                        monitor="val_binary_crossentropy", verbose=0,
                                                        save_best_only=True, save_weights_only=True)]

        model = create_model(input_size=X_train.shape[1], num_trees=trees, trainable=tile_trainable,
                             weights_initializer=layers_weights)
        fit_history = model.fit(x=train_set, batch_size=bs, epochs=tile_epochs, validation_data=valid_set,
                                validation_batch_size=bs, callbacks=callbacks, verbose=verbose)
        print_fit_history(fit_history, epochs)

        model.load_weights("checkpoints/ktree_checkpoint")
        evaluate_history = model.evaluate(x=test_set, batch_size=bs, verbose=0)
        print_evaluate_history(evaluate_history)

        (loss_tileTree[i], acc_tileTree[i]) = evaluate_history[1:]
        model.save_weights(save_folder + f"tile_rc_{trainable_filename}_t{i + 1}_weights")
        np.save(acc_tileTree_filename, acc_tileTree, allow_pickle=True)
        np.save(loss_tileTree_filename, loss_tileTree, allow_pickle=True)
