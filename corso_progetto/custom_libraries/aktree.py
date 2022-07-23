import math
import random
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


def create_asymmetric_tree_structure(target_leaves, asymmetry_index=.2):
    leaves = 0
    layers = [[0, 0]]
    while leaves < target_leaves:

        layers.append([])
        leaves_placer = create_layer(layers[-2].count(0), asymmetry_index)
        for i in range(len(layers[-2])):
            if layers[-2][i] == 0:
                if leaves_placer.pop(0):
                    layers[-2][i] = 1
                    leaves += 1
                    layers[-1].append(-1)
                    layers[-1].append(-1)

                else:
                    layers[-1].append(0)
                    layers[-1].append(0)
            else:
                layers[-1].append(-1)
                layers[-1].append(-1)

        free_nodes = sum(x == 0 for x in layers[-1])
        missing_leaves = target_leaves - leaves
        if free_nodes == missing_leaves:
            layers[-1] = [x if x != 0 else 1 for x in layers[-1]]
            leaves = target_leaves
        elif free_nodes * 2 > missing_leaves:
            layers.append([-1] * (2 * len(layers[-1])))
            free_nodes_indexes = [i for i, e in enumerate(layers[-2]) if e == 0]
            layers[-2] = [x if x != 0 else 1 for x in layers[-2]]
            random.shuffle(free_nodes_indexes)
            while free_nodes < missing_leaves:
                new_node = free_nodes_indexes.pop(0)
                layers[-1][new_node * 2] = 1
                layers[-1][new_node * 2 + 1] = 1
                layers[-2][new_node] = 0
                leaves += 2
                free_nodes -= 1
                missing_leaves = target_leaves - leaves
            leaves = target_leaves

    assign_pixels(layers, 0, 0, 1)
    fill_tree(layers, 0, 0)
    fill_tree(layers, 1, 0)
    print("Depth of the tree: ", len(layers))
    return layers


def assign_pixels(layers, index, depth, next_pixel):
    if depth == 0:
        if layers[0][0] == 1:
            layers[0][0] = next_pixel
            next_pixel += 1
        else:
            next_pixel = assign_pixels(layers, 0, 1, next_pixel)
        if layers[0][1] == 1:
            layers[0][1] = next_pixel
            next_pixel += 1
        else:
            next_pixel = assign_pixels(layers, 1, 1, next_pixel)
    else:
        if layers[depth][2 * index] == 1:
            layers[depth][2 * index] = next_pixel
            next_pixel += 1
        elif layers[depth][2 * index] == 0:
            next_pixel = assign_pixels(layers, 2 * index, depth + 1, next_pixel)
        if layers[depth][2 * index + 1] == 1:
            layers[depth][2 * index + 1] = next_pixel
            next_pixel += 1
        elif layers[depth][2 * index + 1] == 0:
            next_pixel = assign_pixels(layers, 2 * index + 1, depth + 1, next_pixel)

    return next_pixel


def fill_tree(layers, index, depth):
    if depth < len(layers) - 1:
        if layers[depth][index] > 0:
            layers[depth + 1][2 * index] = layers[depth][index]
            layers[depth + 1][2 * index + 1] = layers[depth][index]
        if depth < len(layers) - 2:
            fill_tree(layers, 2 * index, depth + 1)
            fill_tree(layers, 2 * index + 1, depth + 1)


def create_layer(length, asymmetry_index=.2, allow_leaves_only=False):
    layer = []
    while not allow_leaves_only and False not in layer:
        layer = [random.random() < asymmetry_index for _ in range(length)]
    return layer


class AsymmetricTreeLayer(tf.keras.layers.Layer):

    def __init__(self, layer_structure, use_bias=False, non_neg=False):

        super(AsymmetricTreeLayer, self).__init__()
        self.bias = None
        self.masker = None
        self.kernel = None
        self.copy_placer = None
        self.kernel_placer = None
        self.layer_structure = layer_structure
        self.use_bias = use_bias
        self.non_neg = non_neg
        self.num_weights = 0

    def build(self, input_shape):

        self.num_weights = len([i for i in self.layer_structure if self.layer_structure.count(i) == 1 or i == 0])
        placeholder = [e == 0 or self.layer_structure.count(e) == 1 for i, e in enumerate(self.layer_structure) if
                       i == 0 or self.layer_structure[i - 1] != e or e == 0]

        self.masker = np.zeros((len(placeholder), len(placeholder) - self.num_weights // 2))
        masker_counter = 0
        for i, e in enumerate(placeholder):
            self.masker[i, i - masker_counter] = 1.
            if e and i < len(placeholder) - 1 and placeholder[i + 1] and self.masker[i - 1, i - masker_counter] == 0.:
                masker_counter += 1

        if self.non_neg:
            constraint = tf.keras.constraints.NonNeg()
        else:
            constraint = None

        self.kernel = self.add_weight(shape=(1, self.num_weights),
                                      initializer=tf.keras.initializers.HeNormal,
                                      trainable=True, dtype=tf.float32,
                                      constraint=constraint,
                                      name='kernel')

        self.kernel_placer = np.zeros([self.num_weights, len(placeholder)])
        self.copy_placer = np.zeros([1, len(placeholder)])
        kernel_counter = 0
        for i, e in enumerate(placeholder):
            if e:
                self.kernel_placer[kernel_counter, i] = 1
                kernel_counter += 1
            else:
                self.copy_placer = 1.

        self.copy_placer = tf.convert_to_tensor(self.copy_placer, dtype=tf.float32)
        self.kernel_placer = tf.sparse.from_dense(tf.convert_to_tensor(self.kernel_placer, dtype=tf.float32))

        self.masker = tf.convert_to_tensor(self.masker, dtype=tf.float32)
        self.masker = tf.sparse.from_dense(self.masker)

        # TODO: implement biasing for asymmetric tree
        # if self.use_bias:
        #     self.bias = self.add_weight(shape=(self.Input_size // self.divisor,),
        #                                 initializer=tf.keras.initializers.Zeros,
        #                                 trainable=True, name="bias")

    def call(self, inputs):

        multiplier = tf.sparse.sparse_dense_matmul(self.kernel, self.kernel_placer)
        x = tf.nn.leaky_relu(tf.math.multiply(inputs, multiplier), alpha=.01) + tf.math.multiply(inputs,
                                                                                                 self.copy_placer)
        x = tf.sparse.sparse_dense_matmul(x, self.masker)

        # TODO: see above
        # if self.use_bias:
        #     x = tf.nn.bias_add(x, self.bias)

        return x


def create_asymmetric_model(unique_pixels, asymmetry_index=.2, use_bias=False, non_neg=False, learning_rate=1e-03, epsilon=1e-08, layer_structure=None):

    model = tf.keras.Sequential()
    if layer_structure is None:
        layer_structure = create_asymmetric_tree_structure(unique_pixels, asymmetry_index=asymmetry_index)
        layer_structure.reverse()
    for layer in layer_structure:
        model.add(AsymmetricTreeLayer(layer, use_bias=use_bias, non_neg=non_neg))

    if non_neg:
        constraint = tf.keras.constraints.NonNeg()
    else:
        constraint = None

    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid', use_bias=use_bias, kernel_constraint=constraint))

    # if math.log2(len(layer_structure[0])) == math.log2(unique_pixels) + 4:
    #     learning_rate = 1e-2
    # elif math.log2(len(layer_structure[0])) > math.log2(unique_pixels) + 4:
    #     learning_rate = .1
    # else:
    #     learning_rate = 1e-3

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon),
                  loss=tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.AUTO),
                  metrics=['binary_crossentropy', 'acc'])

    return model, len(layer_structure) + 1


def display_depth(tree_structure, num_pixels):
    image = np.zeros([int(num_pixels**.5), int(num_pixels**.5)])
    for i in range(num_pixels):
        found = False
        for j, layer in enumerate(tree_structure):
            if layer.count(i+1) and not found:
                image[int(i//(num_pixels**.5)), int(i - (i//(num_pixels**.5))*(num_pixels**.5))] = j
                found = True
    print(np.min(image), np.max(image))
    hm = plt.imshow(image, cmap="Blues", interpolation=None)
    plt.colorbar(hm)
    plt.show()


