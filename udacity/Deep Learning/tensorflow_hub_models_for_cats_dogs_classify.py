from __future__ import absolute_import, division, print_function

import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras import layers

import numpy as np
import PIL.Image as Image

IMAGE_RES = 224


def format_image(image, label):
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255
    return image, label


if __name__ == '__main__':
    CLASSIFIER_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"

    IMAGE_RES = 224

    model = tf.keras.Sequential([
        hub.KerasLayer(CLASSIFIER_URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))
    ])

    splits = tfds.Split.ALL.subsplit(weighted=(80, 20))

    splits, info = tfds.load('cats_vs_dogs', with_info=True, as_supervised=True, split=splits)

    (train_examples, validation_example) = splits

    num_example = info.splits['train'].num_examples
    num_classes = info.features['label'].num_classes

    BATCH_SIZE = 32

    train_batches = train_examples.shuffle(num_example//4).map(format_image).batch(BATCH_SIZE).prefetch(1)
    validation_batches = validation_example.map(format_image).batch(BATCH_SIZE).prefetch(1)

    image_batch, label_batch = next(iter(train_batches.take(1)))
    image_batch = image_batch.numpy()
    label_batch = label_batch.numpy()

    result_batch = model.predict(image_batch)

    labels_path = tf.keras.utils.get_file('ImageNetLabels.txt',
                                          'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    imagenet_labels = np.array(open(labels_path).read().splitlines())

    predicted_class_names = imagenet_labels[np.argmax(result_batch, axis=-1)]

    plt.figure()
    for n in range(30):
        plt.subplot(6, 5, n + 1)
        plt.imshow(image_batch[n])
        plt.title(predicted_class_names[n])
        plt.axis('off')
    plt.suptitle("ImageNet prediction")
    plt.show()