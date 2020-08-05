import os
import numpy as np
import glob
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

IMAGE_RES = 224
BATCH_SIZE = 32
EPOCHS = 5


def resize_image(image, label):
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES)) / 255
    return image, label


def normalize(images, labes):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labes


def plot_images(image_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(image_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


def create_model():
    CLASSIFIER_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"

    feature_extractor = hub.KerasLayer(CLASSIFIER_URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))
    feature_extractor.trainable = False

    model = tf.keras.Sequential([
        feature_extractor,
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    print(model.summary())

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Loss')
    plt.savefig('./foo.png')
    plt.show()


if __name__ == '__main__':
    splits = tfds.Split.ALL.subsplit(weighted=(70, 30))

    (train, validation), info = tfds.load('tf_flowers', with_info=True, as_supervised=True, split=splits)

    print(info.splits)

    num_classes = info.features['label'].num_classes
    num_train_examples = 0
    num_validation_examples = 0

    for example in train:
        num_train_examples += 1

    num_validation_examples = info.splits['train'].num_examples - num_train_examples

    print('Total Number of Classes: {}'.format(num_classes))
    print('Total Number of Training Images: {}'.format(num_train_examples))
    print('Total Number of Validation Images: {} \n'.format(num_validation_examples))

    for i, example in enumerate(train.take(5)):
        print("image {} shape: {} label: {}".format(i + 1, example[0].shape, example[1]))

    train_batches = train.shuffle(info.splits['train'].num_examples//4).map(resize_image).batch(BATCH_SIZE).prefetch(1)
    validation_batches = validation.map(resize_image).batch(BATCH_SIZE).prefetch(1)

    model = create_model()
    print(model.summary())

    history = model.fit(
        train_batches,
        epochs=EPOCHS,
        validation_data=validation_batches,
    )

    model.save('saved_models\\Classify_Flowers_Exercise_Tranfer_Learning_MobileNet_Flowers')
    plot_history(history)

    class_names = np.array(info.features['label'].names)
    print(class_names)

    image_batch, label_batch = next(iter(train_batches.take(1)))
    image_batch = image_batch.numpy()
    label_batch = label_batch.numpy()

    predicted_batch = model.predict(image_batch)
    predicted_batch = tf.squeeze(predicted_batch).numpy()
    predicted_ids = np.argmax(predicted_batch, axis=-1)
    predicted_class_names = class_names[predicted_ids]

    plt.figure(figsize=(10, 9))
    for n in range(30):
        plt.subplot(6, 5, n + 1)
        plt.imshow(image_batch[n])
        color = 'blue' if predicted_ids[n] == label_batch[n] else "red"
        plt.title(predicted_class_names[n].title(), color=color)
        plt.axis('off')
    plt.suptitle("ImageNet prediction")
    plt.show()
