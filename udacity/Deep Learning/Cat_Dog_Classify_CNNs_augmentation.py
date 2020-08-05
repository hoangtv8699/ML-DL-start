from __future__ import absolute_import, division, print_function

# import tensorflow and tensorflow dataset
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# helper library
import os
import numpy as np
import matplotlib.pyplot as plt


def normalize(images, labes):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labes


def plot_image(image_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(image_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == '__main__':
    _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

    path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)

    base_dir = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    train_cats_dir = os.path.join(train_dir, 'cats')  # directory with my training cats pictures
    train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with my training dogs pictures
    validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with my training cats pictures
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with my training dogs pictures

    num_cats_tr = len(os.listdir(train_cats_dir))
    num_dogs_tr = len(os.listdir(train_dogs_dir))

    num_cats_val = len(os.listdir(validation_cats_dir))
    num_dogs_val = len(os.listdir(validation_dogs_dir))

    total_train = num_cats_tr + num_dogs_tr
    total_val = num_cats_val + num_dogs_val

    print('total training cat images:', num_cats_tr)
    print('total training dog images:', num_dogs_tr)

    print('total validation cat images:', num_cats_val)
    print('total validation dog images:', num_dogs_val)
    print('--')
    print('total train images:', total_train)
    print('total validation images:', total_val)

    BATCH_SIZE = 100
    IMG_SIZE = 150

    # # flip horizontal
    # train_image_generator = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)
    # train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
    #                                                            directory=train_dir,
    #                                                            shuffle=True,
    #                                                            target_size=(IMG_SIZE, IMG_SIZE))
    #
    # augmented_images = [train_data_gen[0][0][0] for i in range(5)]
    # plot_image(augmented_images)

    # # rotating 45 degree
    # train_image_generator = ImageDataGenerator(rescale=1. / 255, rotation_range=45)
    # train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
    #                                                            directory=train_dir,
    #                                                            shuffle=True,
    #                                                            target_size=(IMG_SIZE, IMG_SIZE))
    #
    # augmented_images = [train_data_gen[0][0][0] for i in range(5)]
    # plot_image(augmented_images)

    # # zooming
    # train_image_generator = ImageDataGenerator(rescale=1. / 255, zoom_range=0.5)
    # train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
    #                                                            directory=train_dir,
    #                                                            shuffle=True,
    #                                                            target_size=(IMG_SIZE, IMG_SIZE))
    #
    # augmented_images = [train_data_gen[0][0][0] for i in range(5)]
    # plot_image(augmented_images)

    # combine all augmentation
    train_image_generator = ImageDataGenerator(rescale=1. / 255,
                                               rotation_range=40,
                                               width_shift_range=0.2,
                                               height_shift_range=0.2,
                                               shear_range=0.2,
                                               zoom_range=0.2,
                                               horizontal_flip=True,
                                               fill_mode='nearest')
    train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                               directory=train_dir,
                                                               shuffle=True,
                                                               target_size=(IMG_SIZE, IMG_SIZE),
                                                               class_mode='binary')

    # augmented_images = [train_data_gen[0][0][0] for i in range(5)]
    # plot_image(augmented_images)

    validation_image_generator = ImageDataGenerator(rescale=1. / 255)
    val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                                  directory=validation_dir,
                                                                  shuffle=False,
                                                                  target_size=(IMG_SIZE, IMG_SIZE),
                                                                  class_mode='binary')
    # # plot image test
    # # sample_training_images, _ = next(train_data_gen)
    # # plot_image(sample_training_images[:5])

    model = create_model()
    print(model.summary())

    EPOCHS = 100
    history = model.fit(
        train_data_gen,
        steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
        epochs=EPOCHS,
        validation_data=val_data_gen,
        validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
    )

    model.save('saved_models\\cat_dog_classify_cnn_augmentation')

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
