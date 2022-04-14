import math
import numpy as np
import tensorflow as tf


def init_weights(masker):
    density = np.count_nonzero(masker == 1) / masker.size
    stddev = math.sqrt(2 / (masker.shape[0] * density))
    weights = np.random.normal(size=masker.shape, loc=0., scale=stddev)
    weights[np.where(masker == 0)] = 0
    weights = np.reshape(weights, [-1])
    weights = weights[np.where(weights != 0)]
    return weights


class treeLayer(tf.keras.layers.Layer):

    def __init__(self, Input_size=3072, use_bias=False, divisor=2, non_neg=False):
        super(treeLayer, self).__init__()
        self.bias = None
        self.summer = None
        self.kernel = None
        self.Input_size = Input_size
        self.use_bias = use_bias
        self.divisor = divisor
        self.non_neg = non_neg

    def build(self, input_shape):

        self.summer = np.zeros((self.Input_size, self.Input_size // self.divisor))
        for i in range(self.Input_size):
            self.summer[i, i // self.divisor] = 1

        initializer = init_weights(self.summer)
        initializer = tf.keras.initializers.Constant(initializer)

        if self.non_neg:
            constraint = tf.keras.constraints.NonNeg
        else:
            constraint = None

        self.kernel = self.add_weight(shape=(1, self.Input_size),
                                      initializer=initializer,
                                      trainable=True, dtype=tf.float32,
                                      constraint=constraint,
                                      name='kernel')

        self.summer = tf.convert_to_tensor(self.summer, dtype=tf.float32)
        self.summer = tf.sparse.from_dense(self.summer)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.Input_size // self.divisor,),
                                        initializer=tf.keras.initializers.Zeros,
                                        trainable=True, name="bias")

    def call(self, inputs):
        #print(inputs.shape, self.kernel.shape)
        x = tf.math.multiply(inputs, self.kernel)
        x = tf.sparse.sparse_dense_matmul(x, self.summer)
        #masked_weights = tf.multiply(self.kernel, self.summer)
        #x = tf.matmul(inputs, masked_weights)
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)

        x = tf.nn.leaky_relu(x, alpha=.01)
        return x


def create_model(input_size, num_trees=1, use_bias=False, non_neg=False):
    model = tf.keras.Sequential()
    while input_size > num_trees:

        # Check if image is 3-channel
        if input_size % 3 == 0:
            divisor = 3
        else:
            divisor = 2

        model.add(treeLayer(input_size, use_bias=use_bias, divisor=divisor, non_neg=non_neg))
        input_size = input_size // divisor

    if non_neg:
        constraint = tf.keras.constraints.NonNeg
    else:
        constraint = None

    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid', use_bias=use_bias, kernel_constraint=constraint))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-08),
                  loss=tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.AUTO),
                  metrics=['binary_crossentropy', 'acc'])

    return model
