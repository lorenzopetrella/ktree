import sys
from math import floor


def progress_bar(executed, max=100, pre="", post=""):
    sys.stdout.write('\r')
    sys.stdout.write(
        (str(pre) + "[%-20s] %d%%" + str(post)) % ('=' * floor(executed / max * 20), floor(100 * executed / max)))
    sys.stdout.flush()


def print_fit_history(fit_history, epochs):
    n_epochs = len(fit_history.history['acc'])
    train_loss = round(fit_history.history['loss'][-1] * 100, 2)
    train_acc = round(fit_history.history['acc'][-1] * 100, 2)
    valid_loss = round(fit_history.history['val_loss'][-1] * 100, 2)
    valid_acc = round(fit_history.history['val_acc'][-1] * 100, 2)
    print(
        f"Epochs: {n_epochs}/{epochs} - Train loss: {train_loss}%, accuracy: {train_acc}% - Validation loss: {valid_loss}%, accuracy: {valid_acc}%")


def print_evaluate_history(evaluate_history):
    print(
        f"Test loss: {round(evaluate_history[0] * 100, 2)}%, accuracy: {round(evaluate_history[1] * 100, 2)}%")
