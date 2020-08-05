from __future__ import absolute_import, division, print_function

import pathlib

import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0: print('')
        print('.', end='')


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'], label='Val Error')
    plt.legend()
    plt.ylim([0, 5])
    plt.show()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'], label='Val Error')
    plt.legend()
    plt.ylim([0, 20])
    plt.show()


if __name__ == '__main__':
    dataset_path = keras.utils.get_file("auto-mpg.data",
                                        "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
    print(dataset_path)

    column_names = ['MPG', 'Cylinders', 'Displayment', 'Horsepower',
                    'Weight', 'Acceleration', 'Model Year', 'Origin']
    raw_dataset = pd.read_csv(dataset_path, names=column_names,
                              na_values="?", comment='\t',
                              sep=" ", skipinitialspace=True)

    dataset = raw_dataset.copy()
    print(dataset.tail())

    print(dataset.isna().sum())

    dataset = dataset.dropna()

    origin = dataset.pop('Origin')
    dataset['USA'] = (origin == 1) * 1.0
    dataset['Europe'] = (origin == 2) * 1.0
    dataset['Japan'] = (origin == 3) * 1.0
    print(dataset.tail())

    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    # # plot pair
    # sns.pairplot(train_dataset[["MPG", "Cylinders", "Displayment", "Weight"]], diag_kind="kde")
    # plt.show()

    train_stats = train_dataset.describe()
    train_stats.pop("MPG")
    train_stats = train_stats.transpose()
    print(train_stats)

    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')

    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)

    model = build_model()

    print(model.summary())

    # test model
    example_batch = normed_train_data[0:10]
    example_result = model.predict(example_batch)
    print(example_result)

    # # train model
    # EPOCHS = 1000
    #
    # early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    #
    # history = model.fit(
    #     normed_train_data, train_labels,
    #     epochs=EPOCHS, validation_split=0.2, verbose=0,
    #     callbacks=[early_stop, PrintDot()]
    # )
    # # for early stop
    # model.save('saved_models\\miles_per_gallon_early_stop')
    # # for non early stop
    # model.save('saved_models\\miles_per_gallon_overfitting')

    # plot_history(history)

    # load model
    model = keras.models.load_model('saved_models/miles_per_gallon_early_stop')

    loss, mse, mae = model.evaluate(normed_test_data, test_labels, verbose=0)
    print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

    test_prediction = model.predict(normed_test_data).flatten()

    # plot the test true value and test prediction
    plt.scatter(test_labels, test_prediction)
    plt.xlabel("True Value [MPG]")
    plt.ylabel("Predict Value [MPG]")
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    plt.plot([-100, 100], [-100, 100])
    plt.show()

    # plot Prediction Error
    error = test_prediction - test_labels
    plt.hist(error, bins=25)
    plt.xlabel("Prediction Error [MPG]")
    plt.ylabel("Count")
    plt.show()


