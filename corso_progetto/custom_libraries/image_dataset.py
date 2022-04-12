import math

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class ImageDataset:

    def __init__(self, ds_name, train_test, shuffle_files=True, data_dir="./data", USPS_dir="./USPS/"):

        if ds_name == 'mnist':

            ds = tfds.load('mnist', split=train_test, shuffle_files=shuffle_files,
                           download=True, data_dir=data_dir, as_supervised=True, batch_size=-1)

        elif ds_name == 'fmnist':

            ds = tfds.load('fashion_mnist', split=train_test, shuffle_files=shuffle_files,
                           download=True, data_dir=data_dir, as_supervised=True, batch_size=-1)

        elif ds_name == 'cifar10':

            ds = tfds.load('cifar10', split=train_test, shuffle_files=shuffle_files,
                           download=True, data_dir=data_dir, as_supervised=True, batch_size=-1)

        elif ds_name == 'kmnist':

            ds = tfds.load('kmnist', split=train_test, shuffle_files=shuffle_files,
                           download=True, data_dir=data_dir, as_supervised=True, batch_size=-1)

        elif ds_name == 'emnist':

            ds = tfds.load('emnist', split=train_test, shuffle_files=shuffle_files,
                           download=True, data_dir=data_dir, as_supervised=True, batch_size=-1)

        elif ds_name == 'svhn':

            ds = tfds.load('svhn_cropped', split=train_test, shuffle_files=shuffle_files,
                           download=True, data_dir=data_dir, as_supervised=True, batch_size=-1)

        elif ds_name == 'usps':

            self.images = np.load(USPS_dir + train_test + '_x.npy')
            self.labels = np.load(USPS_dir + train_test + '_y.npy')

        else:
            raise Exception("Selected dataset is not available")

        if ds_name != 'usps':
            self.images, self.labels = tfds.as_numpy(ds)

        self.num_channels = self.images.shape[3]

    def shuffle(self):

        p = np.random.permutation(len(self.images))
        self.images = self.images[p]
        self.labels = self.labels[p]

    def normalize(self, n_bits=8):

        self.images = self.images / (2 ** n_bits - 1)

    def filter(self, t1, t2=None, binary=True, overwrite=False):

        images = self.images
        labels = self.labels
        target_1 = np.where(labels == t1)
        target_2 = np.where(labels == t2)

        if t2 is not None:
            images = images[np.where((labels == t1) | (labels == t2))]
            labels = labels[np.where((labels == t1) | (labels == t2))]

        target_1 = np.where(labels == t1)
        target_2 = np.where(labels == t2)

        if binary:
            labels[target_1] = 1
            labels[target_2] = 0

        if not overwrite:
            return images, labels

        else:
            self.images = images
            self.labels = labels

    def pad(self, target_shape=(32, 32)):
        """
        Adds symmetric zero padding to image with shape HxWxC
        :param target_shape: tuple with target shape (H,W)
        """

        self.images = np.pad(self.images, ((0, 0), (2, 2), (2, 2), (0, 0)))

    def vectorize(self, merge_channels=False, by_row=True):
        """
        Transforms image from pixel matrix with shape (H,W,C) to linear vector
        :param merge_channels: if True, the pixel vectors from each channel will be concatenated
        :param by_row: if True, concatenate pixels by row, otherwise by column
        """

        if by_row:
            self.images = np.transpose(self.images, (0, 3, 1, 2))
        else:
            self.images = np.transpose(self.images, (0, 3, 2, 1))

        if merge_channels:
            self.images = np.reshape(self.images, [len(self.images), -1])

        else:
            self.images = np.reshape(self.images, [len(self.images), self.images.shape[1], -1])

    def permute(self, n):

        perm_idx = get_permutation(n)
        if self.num_channels == 3:
            perm_idx = np.concatenate((perm_idx, perm_idx, perm_idx), 0)
        self.images = self.images[:, perm_idx]

    def subset(self, shard=False, shard_number=None, validation=False, validation_size=None, tile=None):

        if validation and validation_size is None:
            raise Exception("Requested training/validation split but no validation split size given.")
        if shard and shard_number is None:
            raise Exception("Requested sharding but no shard size given.")

        if validation:
            valid_images, valid_labels = self.images[0:validation_size], self.labels[0:validation_size]
            train_images, train_labels = self.images[validation_size:], self.labels[validation_size:]
            if tile is not None:
                valid_images = np.tile(valid_images, (1, tile))
                train_images = np.tile(train_images, (1, tile))
            if shard:
                valid_images, valid_labels = np.array_split(valid_images, shard_number), np.array_split(valid_labels,
                                                                                                        shard_number)
                train_images, train_labels = np.array_split(train_images, shard_number), np.array_split(train_labels,
                                                                                                        shard_number)
            return train_images, train_labels, valid_images, valid_labels
        else:
            train_images, train_labels = self.images, self.labels
            if tile is not None:
                train_images = np.tile(train_images, (1, tile))
            if shard:
                train_images, train_labels = np.array_split(train_images, shard_number), np.array_split(train_labels,
                                                                                                        shard_number)
            return train_images, train_labels

    def bootstrap(self, train_size, validation=False):

        train_size = math.floor(train_size*len(self.images))

        self.shuffle()
        train_images = self.images[0:train_size]
        train_labels = self.labels[0:train_size]
        valid_images = self.images[train_size:]
        valid_labels = self.labels[train_size:]

        if validation:
            return train_images, train_labels, valid_images, valid_labels
        else:
            return train_images, train_labels


def get_matrix(n):
    '''
     Assumes that the matrix is of size 2^n x 2^n for some n

     EXAMPLE for n=4

     Old order:

      1  2  3  4
      5  6  7  8
      9 10 11 12
     13 14 15 16

     New order:

      1  2  5  6
      3  4  7  8
      9 10 13 14
     11 12 15 16

     Function returns numbers from old order, read in the order of the new numbers:

     [1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16]

     So if you previously had a data vector v from a matrix size 32 x 32,
     you can now use v[get_permutation(5)] to reorder the elements.
    '''
    if n == 0:
        return np.array([[1]])
    else:
        smaller = get_matrix(n - 1)
        num_in_smaller = 2 ** (2 * n - 2)
        first_stack = np.hstack((smaller, smaller + num_in_smaller))
        return np.vstack((first_stack, first_stack + 2 * num_in_smaller))


def get_permutation(n):
    return get_matrix(n).ravel() - 1
