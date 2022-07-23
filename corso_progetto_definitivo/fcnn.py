from custom_libraries.image_dataset import *
from custom_libraries.miscellaneous import *
from custom_libraries.fcnn import create_model
from os.path import exists
import numpy as np

#############
### SETUP ###
#############

project_folder = ''
bs = 256
trials = 10
epochs = 2000
trees_set = [1, 32]
classes = np.load(project_folder + 'results/classes.npy', allow_pickle=True)
save_folder = project_folder + "results/fcnn/"

##############
### /SETUP ###
##############

for j, (t1, t2, ds) in enumerate(classes):

    print(f"Dataset: {ds} / Pair: {t1}-{t2}")

    test_ds = ImageDataset(ds, 'test', data_dir=None, USPS_dir=project_folder + 'USPS/')
    train_ds = ImageDataset(ds, 'train', data_dir=None, USPS_dir=project_folder + 'USPS/')

    for x in [train_ds, test_ds]:
        x.filter(t1, t2, overwrite=True)
        x.shuffle()
        x.normalize()
        if x.images.shape[1:3] == (28, 28):
            x.pad()
        x.vectorize(True)

    test_set = tf.data.Dataset.from_tensor_slices((test_ds.images, test_ds.labels)).batch(bs)

    for k, trees in enumerate(trees_set):

        filename_acc = save_folder + f"fcnn{trees}_{ds}_acc.npy"
        filename_loss = save_folder + f"fcnn{trees}_{ds}_loss.npy"
        if exists(filename_acc) and exists(filename_loss):
            acc = np.load(filename_acc, allow_pickle=True)
            loss = np.load(filename_loss, allow_pickle=True)
        else:
            acc = np.zeros(trials)
            loss = np.zeros(trials)

        print(f"{trees}-FCNN")

        X_train, y_train, X_valid, y_valid = train_ds.subset(shard=True, shard_number=trials, validation=True,
                                                             validation_size=len(test_ds.labels))

        for i in range(trials):

            if acc[i] != 0:
                continue

            print(f"Trial {i + 1}")

            with tf.device('/device:GPU:0'):

                model = create_model(num_units=2 * trees)

                train_set = tf.data.Dataset.from_tensor_slices((X_train[i], y_train[i])).batch(bs)
                valid_set = tf.data.Dataset.from_tensor_slices((X_valid[i], y_valid[i])).batch(bs)

                callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=60),
                             tf.keras.callbacks.ModelCheckpoint(filepath="checkpoints/fcnn_checkpoint",
                                                                monitor='val_binary_crossentropy',
                                                                verbose=0,
                                                                save_best_only=True,
                                                                save_weights_only=True,
                                                                )]
                fit_history = model.fit(x=train_set, batch_size=bs, epochs=epochs,
                                        validation_data=valid_set, validation_batch_size=bs,
                                        callbacks=callbacks, verbose=0)
                print_fit_history(fit_history, epochs)
                model.load_weights('checkpoints/fcnn_checkpoint')

                evaluate_history = model.evaluate(x=test_set, batch_size=bs, verbose=0)
                print_evaluate_history(evaluate_history)

                (loss[i], acc[i]) = evaluate_history[1:]

                np.save(filename_acc, acc, allow_pickle=True)
                np.save(filename_loss, loss, allow_pickle=True)
