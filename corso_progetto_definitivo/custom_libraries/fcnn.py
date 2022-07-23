import tensorflow as tf


def create_model(num_units):
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Dense(units=num_units, activation=None, kernel_initializer=tf.keras.initializers.HeNormal))
    model.add(tf.keras.layers.LeakyReLU(alpha=.01))
    model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
                  metrics=['binary_crossentropy', 'acc'])

    return model
