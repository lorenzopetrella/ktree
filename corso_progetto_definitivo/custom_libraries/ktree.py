import math
import numpy as np
import tensorflow as tf
import time
from os.path import exists

def init_weights(masker):
    density = np.count_nonzero(masker == 1) / masker.size
    stddev = math.sqrt(2 / (masker.shape[0] * density))
    weights = np.random.normal(size=masker.shape, loc=0., scale=stddev)
    weights[np.where(masker == 0)] = 0
    weights = np.reshape(weights, [-1])
    weights = weights[np.where(weights != 0)]
    return weights


class treeLayer(tf.keras.layers.Layer):

    def __init__(self, Input_size=3072, use_bias=False, divisor=2, non_neg=False, trainable=True,
                 layer_initializer=None):
        super(treeLayer, self).__init__()
        self.bias = None
        self.summer = None
        self.kernel = None
        self.Input_size = Input_size
        self.use_bias = use_bias
        self.divisor = divisor
        self.non_neg = non_neg
        self.trainable = trainable
        self.layer_initializer = layer_initializer

    def build(self, input_shape):

        self.summer = np.zeros((self.Input_size, self.Input_size // self.divisor))
        for i in range(self.Input_size):
            self.summer[i, i // self.divisor] = 1

        if self.layer_initializer is None:
            initializer = init_weights(self.summer)
            initializer = tf.keras.initializers.Constant(initializer)
        else:
            initializer = tf.keras.initializers.Constant(self.layer_initializer)

        if self.non_neg:
            constraint = tf.keras.constraints.NonNeg()
        else:
            constraint = None

        self.kernel = self.add_weight(shape=(1, self.Input_size),
                                      initializer=initializer,
                                      trainable=self.trainable, dtype=tf.float32,
                                      constraint=constraint,
                                      name='kernel')

        self.summer = tf.convert_to_tensor(self.summer, dtype=tf.float32)
        self.summer = tf.sparse.from_dense(self.summer)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.Input_size // self.divisor,),
                                        initializer=tf.keras.initializers.Zeros,
                                        trainable=self.trainable, name="bias")

    def call(self, inputs):
        # print(inputs.shape, self.kernel.shape)
        x = tf.math.multiply(inputs, self.kernel)
        x = tf.sparse.sparse_dense_matmul(x, self.summer)
        # masked_weights = tf.multiply(self.kernel, self.summer)
        # x = tf.matmul(inputs, masked_weights)
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)

        x = tf.nn.leaky_relu(x, alpha=.01)
        return x


def create_model(input_size, num_trees=1, use_bias=False, non_neg=False, weights_initializer=None, trainable=True):

    model = tf.keras.Sequential()

    if use_bias and weights_initializer is not None:
        raise Warning("Bias initialization has not been implemented")

    while input_size > num_trees:

        # Check if the image is 3-channel (and not just a 3k-tree)
        if input_size // num_trees > 0 and input_size // num_trees % 3 == 0:
            divisor = 3
        else:
            divisor = 2

        if weights_initializer is not None:
            layer_initializer = weights_initializer.pop(0)
        else:
            layer_initializer = None

        model.add(treeLayer(input_size, use_bias=use_bias, divisor=divisor, non_neg=non_neg,
                            layer_initializer=layer_initializer, trainable=trainable))
        input_size = input_size // divisor

    if non_neg:
        constraint = tf.keras.constraints.NonNeg()
    else:
        constraint = None

    if weights_initializer is not None:
        layer_initializer = tf.keras.initializers.Constant(weights_initializer.pop(0))
    else:
        layer_initializer = None

    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid', use_bias=use_bias, kernel_constraint=constraint,
                                    kernel_initializer=layer_initializer))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-08),
                  loss=tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.AUTO),
                  metrics=['binary_crossentropy', 'acc'])

    return model


class TimingCallback(tf.keras.callbacks.Callback):

    def __init__(self, save_file='./', offset_epochs=10):
        if exists(save_file):
            self.time_vector = np.load(save_file, allow_pickle=True)
        else:
            self.time_vector = np.empty(0, dtype=np.float32)
        self.offset_epochs = offset_epochs
        self.save_file = save_file
        self.time_per_step = np.empty(0, dtype=np.float32)
        self._previous_epoch_iterations = None
        self._epoch_start_time = None

    def _compute_steps_per_second(self):
        current_iteration = self.model.optimizer.iterations.numpy()
        time_since_epoch_begin = time.time() - self._epoch_start_time
        steps_per_second = ((current_iteration - self._previous_epoch_iterations) /
                            time_since_epoch_begin)
        return steps_per_second

    def on_epoch_begin(self, epoch, logs=None):
        self._previous_epoch_iterations = self.model.optimizer.iterations.numpy()
        self._epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        time_per_step = self._compute_steps_per_second() ** -1 * 1000
        if epoch >= self.offset_epochs:
            self.time_per_step = np.concatenate((self.time_per_step, [time_per_step]))
        print(f"Epoch {epoch+1}, {time_per_step} ms/step")

    def on_train_end(self, logs=None):
        self.time_vector = np.concatenate((self.time_vector, [np.mean(self.time_per_step)]))
        np.save(self.save_file, self.time_per_step, allow_pickle=True)
        print(f"Average speed: {np.mean(self.time_per_step)} ms/step")