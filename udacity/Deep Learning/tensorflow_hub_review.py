from __future__ import absolute_import, division, print_function

import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras import layers

import numpy as np
import PIL.Image as Image

if __name__ == '__main__':
    CLASSIFIER_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"

    IMAGE_RES = 224

    model = tf.keras.Sequential([
        hub.KerasLayer(CLASSIFIER_URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))
    ])

    grace_hopper = tf.keras.utils.get_file('image.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
    grace_hopper = Image.open(grace_hopper).resize((IMAGE_RES, IMAGE_RES))

    grace_hopper = np.array(grace_hopper) / 255
    print(grace_hopper.shape)

    result = model.predict(grace_hopper[np.newaxis, ...])
    print(result.shape)  # 1D array with 1001 mem

    predicted_class = np.argmax(result)
    print(predicted_class)

    labels_path = tf.keras.utils.get_file('ImageNetLabels.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    imagenet_labels = np.array(open(labels_path).read().splitlines())

    plt.imshow(grace_hopper)
    plt.axis('off')
    predicted_class_name = imagenet_labels[predicted_class]
    plt.title("Prediction: " + predicted_class_name.title())
    plt.show()
