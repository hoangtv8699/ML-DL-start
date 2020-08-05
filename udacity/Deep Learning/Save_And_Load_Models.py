import time
import numpy as np
import  matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

from tensorflow.keras import layers

BATCH_SIZE = 32
EPOCHS = 2
IMAGE_RES = 224
CLASSIFIER_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"


def format_image(image, label):
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES)) / 255.0
    return image, label


def create_model():
    feature_extractor = hub.KerasLayer(CLASSIFIER_URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))
    feature_extractor.trainable = False

    model = tf.keras.Sequential([
        feature_extractor,
        layers.Dense(2, activation='softmax')
    ])

    print(model.summary())

    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model


if __name__ == '__main__':
    splits = tfds.Split.ALL.subsplit(weighted=(80, 20))
    (train_examples, validation_examples), info = tfds.load('cats_vs_dogs', with_info=True, as_supervised=True, split=splits)

    num_examples = info.splits['train'].num_examples
    train_batches = train_examples.cache().shuffle(num_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)
    validation_batches = validation_examples.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)

    # model = create_model()
    #
    # history = model.fit(
    #     train_batches,
    #     epochs=EPOCHS,
    #     validation_data=validation_batches
    # )
    # save model
    # model.save('saved_models\\save_and_load_models')

    # save model .h5
    # t = time.time()
    # save_path_keras = "saved_models_h5\\./{}.h5".format(int(t))
    # model.save(save_path_keras)

    # save model tf format
    # t = time.time()
    # save_path_keras = "saved_models_h5\\./{}".format(int(t))
    # tf.saved_model.save(model, save_path_keras)

    # load model
    # model = tf.saved_model.load('saved_models\\save_and_load_models')

    # load model
    model = tf.keras.models.load_model('saved_models_h5/1595585188.h5',
                                       custom_objects={'KerasLayer': hub.KerasLayer})

    class_names = np.array(info.features['label'].names)
    print(class_names)

    image_batches, label_batches = next(iter(train_batches.take(1)))
    image_batches = image_batches.numpy()
    label_batches = label_batches.numpy()

    predicted_batch = model.predict(image_batches)
    predicted_batch = tf.squeeze(predicted_batch).numpy()
    predicted_ids = np.argmax(predicted_batch, axis=-1)
    predicted_class_names = class_names[predicted_ids]

    plt.figure(figsize=(10, 9))
    for n in range(30):
        plt.subplot(6, 5, n + 1)
        plt.imshow(image_batches[n])
        color = 'blue' if predicted_ids[n] == label_batches[n] else "red"
        plt.title(predicted_class_names[n].title(), color=color)
        plt.axis('off')
    plt.suptitle("ImageNet prediction")
    plt.show()
