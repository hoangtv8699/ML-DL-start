import os
import numpy as np
import glob
import shutil

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

IMAGE_SHAPE = 224
BATCH_SIZE = 32
EPOCHS = 6


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

    feature_extractor = hub.KerasLayer(CLASSIFIER_URL, input_shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3))
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
    _URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

    zip_file = tf.keras.utils.get_file(origin=_URL,
                                       fname="flower_photos.tgz",
                                       extract=True)

    base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')
    daisy_dir = os.path.join(base_dir, 'daisy')
    dandelion_dir = os.path.join(base_dir, 'dandelion')
    roses_dir = os.path.join(base_dir, 'roses')
    sunflowers_dir = os.path.join(base_dir, 'sunflowers')
    tulips_dir = os.path.join(base_dir, 'tulips')

    total_daisy = len(os.listdir(daisy_dir))
    total_dandelion = len(os.listdir(dandelion_dir))
    total_roses = len(os.listdir(roses_dir))
    total_sunflowers = len(os.listdir(sunflowers_dir))
    total_tulips = len(os.listdir(tulips_dir))

    total_data = total_daisy + total_roses + total_dandelion + total_tulips + total_sunflowers

    image_generator = ImageDataGenerator(rescale=1. / 255,
                                         rotation_range=45,
                                         width_shift_range=0.15,
                                         height_shift_range=0.15,
                                         shear_range=0.2,
                                         zoom_range=0.5,
                                         horizontal_flip=True,
                                         fill_mode='nearest',
                                         validation_split=0.2)
    train_data_gen = image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                         directory=base_dir,
                                                         shuffle=True,
                                                         class_mode='sparse',
                                                         target_size=(IMAGE_SHAPE, IMAGE_SHAPE),
                                                         subset='training')
    validation_data_gen = image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=base_dir,
                                                              shuffle=True,
                                                              class_mode='sparse',
                                                              target_size=(IMAGE_SHAPE, IMAGE_SHAPE),
                                                              subset='validation')

    # augmented_images = [train_data_gen[0][0][0] for i in range(5)]
    # plot_images(augmented_images)

    model = create_model()
    print(model.summary())

    history = model.fit(
        train_data_gen,
        steps_per_epoch=int(np.ceil(train_data_gen.n / float(BATCH_SIZE))),
        epochs=EPOCHS,
        validation_data=validation_data_gen,
        validation_steps=int(np.ceil(validation_data_gen.n / float(BATCH_SIZE)))
    )

    model.save('saved_models\\Classify_Flowers_Exercise_Tranfer_Learning_2')
    plot_history(history)
