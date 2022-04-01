import sys
from math import floor

import numpy as np


def progress_bar(executed, max=100, pre="", post=""):
    sys.stdout.write('\r')
    sys.stdout.write(
        (str(pre) + "[%-20s] %d%%" + str(post)) % ('=' * floor(executed / max * 20), floor(100 * executed / max)))
    sys.stdout.flush()


def print_fit_history(fit_history, epochs):

    best_epoch = np.argmin(fit_history.history['val_binary_crossentropy'])
    if best_epoch.size > 1:
        best_epoch = best_epoch[0]

    n_epochs = len(fit_history.history['acc'])
    train_loss = round(fit_history.history['binary_crossentropy'][best_epoch], 4)
    train_acc = round(fit_history.history['acc'][best_epoch] * 100, 2)
    valid_loss = round(fit_history.history['val_binary_crossentropy'][best_epoch], 4)
    valid_acc = round(fit_history.history['val_acc'][best_epoch] * 100, 2)
    print(
        f"Epochs: {n_epochs}/{epochs} (best = {best_epoch+1}) - Train BCE: {train_loss}, accuracy: {train_acc}% - Validation BCE: {valid_loss}, accuracy: {valid_acc}%")


def print_evaluate_history(evaluate_history):
    print(
        f"Test BCE: {round(evaluate_history[1], 4)}, accuracy: {round(evaluate_history[2] * 100, 2)}%")