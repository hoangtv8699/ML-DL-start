from __future__ import absolute_import, division, print_function

# import tensorflow and tensorflow dataset
import tensorflow as tf
import tensorflow_datasets as tfds
assert tf.config.list_physical_devices('GPU')

# helper library
import math
import numpy as np
import matplotlib.pyplot as plt


def normalize(images, labes):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labes


def create_CNNS_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu, input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == '__main__':
    dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']

    class_names = ['T-shirst/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    num_train_example = metadata.splits['train'].num_examples
    num_test_example = metadata.splits['test'].num_examples
    print(num_train_example)
    print(num_test_example)

    # train_dataset = train_dataset.map(normalize)
    # test_dataset = test_dataset.map(normalize)

    model = create_CNNS_model()

    BATCH_SIZE = 32
    train_dataset = train_dataset.repeat().shuffle(num_train_example).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    model.fit(train_dataset, epochs=10, steps_per_epoch=math.ceil(num_train_example / BATCH_SIZE))

    model.save('saved_models\\MNIST_With_CNNs_non_normalize')

    tess_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_example / BATCH_SIZE))
    print('Accuracy on test dataset:', test_accuracy)