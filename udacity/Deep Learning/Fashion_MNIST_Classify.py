# import Tensorflow, Tensorflow Datasets
import tensorflow as tf
import tensorflow_datasets as tfds

# helper libraries
import math
import numpy as np
import matplotlib.pyplot as plt

# Improve progessbar display
import tqdm
import tqdm.auto

tqdm.tqdm = tqdm.auto.tqdm


def normalize(images, labes):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labes


def plot_image(i, predictions_array, true_labels, images):
    predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img[..., 0], cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.argmax(predictions_array),
                                         class_names[true_label],
                                         color=color))


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def create_model():
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

    # take 1 image and remove the color dimension by reshaping
    for image, label in test_dataset.take(1):
        break
    image = image.numpy().reshape((28, 28))

    # # plot the image
    # plt.figure()
    # plt.imshow(image, cmap=plt.cm.binary)
    # plt.colorbar()
    # plt.grid(False)
    # plt.show()
    #
    # # plot 25 image
    # plt.figure(figsize=(10, 10))
    # i = 0
    # for (image, label) in test_dataset.take(25):
    #     image = image.numpy().reshape((28, 28))
    #     plt.subplot(5, 5, i + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(image, cmap=plt.cm.binary)
    #     plt.xlabel(class_names[label])
    #     i += 1
    # plt.show()

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    BATCH_SIZE = 32
    train_dataset = train_dataset.repeat().shuffle(num_train_example).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(num_train_example / BATCH_SIZE))

    model.save('saved_models\\Fashion_MNIST_Classify_example_not_normalize_pixel')

    tess_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_example / BATCH_SIZE))
    print('Accuracy on test dataset:', test_accuracy)

    # for test_images, test_labels in test_dataset.take(1):
    #     test_images = test_images.numpy()
    #     test_labels = test_labels.numpy()
    #     predictions = model.predict(test_images)
    #     print(predictions.shape)
    #     print(predictions[0])
    #     print(np.argmax(predictions[0]))
    #     print(test_labels[0])

    # i = 12
    # plt.figure(figsize=(6,3))
    # plt.subplot(1,2,1)
    # plot_image(i, predictions, test_labels, test_images)
    # plt.subplot(1,2,2)
    # plot_value_array(i, predictions, test_labels)
    # plt.show()

    return model


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # print(tf.__version__)
    #
    # dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
    # train_dataset, test_dataset = dataset['train'], dataset['test']
    #
    # class_names = ['T-shirst/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    #                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    #
    # num_train_example = metadata.splits['train'].num_examples
    # num_test_example = metadata.splits['test'].num_examples
    # print(num_train_example)
    # print(num_test_example)
    #
    # train_dataset = train_dataset.map(normalize)
    # test_dataset = test_dataset.map(normalize)
    #
    # model = tf.keras.models.load_model('saved_models/Fashion_MNIST_Classify_example')
    #
    # BATCH_SIZE = 25
    # test_dataset = test_dataset.batch(BATCH_SIZE)
    #
    # tess_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_example / BATCH_SIZE))
    # print('Accuracy on test dataset:', test_accuracy)
    # model = create_model()
