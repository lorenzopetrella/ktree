from custom_libraries.miscellaneous import *
from custom_libraries.image_dataset import *
from custom_libraries.aktree_fast import *
import numpy as np
import gc
from os.path import exists
import pickle
import time

fmt = lambda t: time.strftime('%a %b %d %H:%M:%S %Z %Y', t)

#############
### SETUP ###
#############

project_folder = ''
bs = 256
trials = 10
epochs = 2000
patience = 500
trees_set = [1]
asymmetry_index = .3
classes = [[3, 5, 'mnist']]
verbose = 1
learning_rate = 1e-03

##############
### /SETUP ###
##############

devices = tf.config.list_logical_devices()
if len(devices) == 1:
    device_name = devices[0].name
else:
    for device in devices:
        if 'GPU' in device.name:
            device_name = device.name
            break
print('Will run on the following device:', device_name)

with tf.device(device_name):

    for j, (t1, t2, ds) in enumerate(classes):

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

        X_train, y_train, X_valid, y_valid = train_ds.bootstrap(.85, True)

        for k, trees in enumerate(trees_set):

            print(f"{trees}-tree")

            filename_acc = project_folder + f"results/a-{trees}-tree_" + ds + "_" + str(
                int(100 * asymmetry_index)) + '_acc.npy'
            filename_loss = project_folder + f"results/a-{trees}-tree_" + ds + "_" + str(
                int(100 * asymmetry_index)) + '_loss.npy'
            filename_trees = project_folder + f"results/a-{trees}-tree_" + ds + "_" + str(
                int(100 * asymmetry_index)) + '_trees.npy'

            if exists(filename_acc) and exists(filename_loss) and exists(filename_trees):
                print("Recovering data...")
                acc, loss = np.load(filename_acc, allow_pickle=True), np.load(filename_loss, allow_pickle=True)
                with open(filename_trees, "rb") as fp:
                    trees_structure = pickle.load(fp)
            else:
                print("Results will be saved to:", filename_acc, filename_loss, filename_trees)
                acc = np.zeros(trials)
                loss = np.zeros(trials)
                trees_structure = []
                model_weights = []

            train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train)).map(
                lambda x, y: (tf.tile(x, [trees]), y)).batch(bs)
            valid_set = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).map(
                lambda x, y: (tf.tile(x, [trees]), y)).batch(bs)
            test_set = tf.data.Dataset.from_tensor_slices((test_ds.images, test_ds.labels)).map(
                lambda x, y: (tf.tile(x, [trees]), y)).batch(bs)

            for i in range(trials):

                if acc[i] != 0:
                    continue

                print(f"Trial {i + 1}")

                callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=patience),
                             tf.keras.callbacks.ModelCheckpoint(filepath="checkpoints/aktree_orig_checkpoint",
                                                                monitor='val_binary_crossentropy',
                                                                verbose=0,
                                                                save_best_only=True,
                                                                save_weights_only=True,
                                                                )]

                learning_pars = {"initial_learning_rate": 1e-4, "maximal_learning_rate": 1e-2,
                                  "step_size": int(5*len(y_train)/bs), "scale_mode": "cycle"}
                model, new_tree = create_asymmetric_model(unique_pixels=len(train_ds.images[0]),
                                                          use_bias=False, non_neg=False,
                                                          asymmetry_index=asymmetry_index,
                                                          learning_mode="constant", learning_pars=None)

                trees_structure.append(new_tree)

                start = time.localtime()
                print(f"Training started at {fmt(start)}.")
                fit_history = model.fit(x=train_set, batch_size=bs, epochs=epochs,
                                        validation_data=valid_set, validation_batch_size=bs,
                                        callbacks=callbacks, verbose=verbose)
                stop = time.localtime()
                dur = time.mktime(stop) - time.mktime(start)
                print(f"Training {i + 1} ended at {fmt(stop)}. Elapsed time: {dur:.0f} seconds.")

                print_fit_history(fit_history, epochs)
                model.load_weights('checkpoints/aktree_orig_checkpoint')
                model.save_weights(f"results/a-{trees}-tree_" + ds + "_" + str(
                                    int(100 * asymmetry_index)) + f"_weights{i}")

                evaluate_history = model.evaluate(x=test_set, batch_size=bs, verbose=0)
                print_evaluate_history(evaluate_history)

                (loss[i], acc[i]) = evaluate_history[1:]

                np.save(filename_acc, acc, allow_pickle=True)
                np.save(filename_loss, loss, allow_pickle=True)
                with open(filename_trees, "wb") as fp:
                    pickle.dump(trees_structure, fp)

                del model
                gc.collect()
