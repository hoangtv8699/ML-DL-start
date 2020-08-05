import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v2 as tf
import tensorflow.keras as keras


def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel('time')
    plt.ylabel('value')
    if label:
        plt.legend(fontsize=14)
    plt.grid = True


def trend(time, slope=0):
    return time * slope


def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


def moving_average_forecast(series, window_size):
    """Forecast the mean of the last few values.
    if window_size = , then this is equivalent to naive forecast"""
    forecast = []
    for time in range(len(series) - window_size):
        forecast.append(series[time:time + window_size].mean())
    return np.array(forecast)


def moving_average_forecast_fast(series, window_size):
    """Forecast the mean of the last few values.
    if window_size = , then this is equivalent to naive forecast.
    this implementation is much faster than previous one"""
    mov = np.cumsum(series)
    mov[window_size:] = mov[window_size:] - mov[:-window_size]
    return mov[window_size - 1: -1] / window_size


def window_dataset(series, window_size, batch_size=32, shuffle_buffer=1000):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast


def seq2seq_window_dataset(series, window_size, batch_size=32,
                           shuffle_buffer=1000):
    series = tf.expand_dims(series, axis=1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[1:]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


if __name__ == '__main__':
    time = np.arange(4 * 365 + 1)
    baseline = 10
    series = baseline + trend(time, 0.1)

    amplitude = 40

    slope = 0.05
    series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)

    noise_level = 5
    noise = white_noise(time, noise_level, seed=42)

    series += noise
    # plt.figure(figsize=(10, 6))
    # plot_series(time, series)

    split_time = 1000
    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid = time[split_time:]
    x_valid = series[split_time:]

    tf.random.set_seed(42)
    np.random.seed(42)

    window_size = 30
    train_set = seq2seq_window_dataset(x_train, window_size, batch_size=128)
    valid_set = seq2seq_window_dataset(x_valid, window_size, batch_size=128)

    # finding learning rate
    # model = keras.models.Sequential([
    #     keras.layers.Conv1D(
    #         filters=32, kernel_size=5,
    #         strides=1, padding="causal",
    #         activation="relu",
    #         input_shape=[None, 1]
    #     ),
    #     keras.layers.LSTM(100, return_sequences=True),
    #     keras.layers.LSTM(100, return_sequences=True),
    #     keras.layers.Dense(1),
    #     keras.layers.Lambda(lambda x: x * 200.0)
    # ])
    #
    # lr_schedule = keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 30))
    # optimizer = keras.optimizers.SGD(lr=1e-8, momentum=0.9)
    # model.compile(
    #     loss=keras.losses.Huber(),
    #     optimizer=optimizer,
    #     metrics=["mae"]
    # )
    #
    # history = model.fit(
    #     train_set,
    #     epochs=100,
    #     callbacks=[lr_schedule]
    # )
    #
    # plt.semilogx(history.history["lr"], history.history["loss"])
    # plt.axis([1e-8, 1e-4, 0, 30])
    # plt.show()

    # real train
    # model = keras.models.Sequential([
    #     keras.layers.Conv1D(
    #         filters=32, kernel_size=5,
    #         strides=1, padding="causal",
    #         activation="relu",
    #         input_shape=[None, 1]
    #     ),
    #     keras.layers.LSTM(100, return_sequences=True),
    #     keras.layers.LSTM(100, return_sequences=True),
    #     keras.layers.Dense(1),
    #     keras.layers.Lambda(lambda x: x * 200.0)
    # ])
    #
    # early_stopping = keras.callbacks.EarlyStopping(patience=50)
    # model_checkpoint = keras.callbacks.ModelCheckpoint("saved_models\\best_checkpoint_RNNs_CNNs_pre_processing.h5",
    #                                                    save_best_only=True)
    # optimizer = keras.optimizers.SGD(lr=5e-6, momentum=0.9)
    # model.compile(
    #     loss=keras.losses.Huber(),
    #     optimizer=optimizer,
    #     metrics=["mae"]
    # )
    #
    # history = model.fit(
    #     train_set,
    #     epochs=500,
    #     validation_data=valid_set,
    #     callbacks=[early_stopping, model_checkpoint]
    # )
    #
    # model = keras.models.load_model('saved_models\\best_checkpoint_RNNs_CNNs_pre_processing.h5')
    #
    # rnn_forecast = model_forecast(model,
    #                               series[..., np.newaxis],
    #                               window_size=window_size)
    # rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]
    # plt.figure(figsize=(10, 6))
    # plot_series(time_valid, x_valid)
    # plot_series(time_valid, rnn_forecast)
    # plt.show()
    #
    # print(keras.metrics.mean_absolute_error(x_valid, rnn_forecast))

    # fully Convonutional model
    # finding learning rate
    # model = keras.models.Sequential()
    # model.add(keras.layers.InputLayer(input_shape=[None, 1]))
    # for dalation_rate in (1, 2, 4, 8, 16, 32):
    #     model.add(
    #         keras.layers.Conv1D(
    #             filters=32, kernel_size=2,
    #             strides=1, dilation_rate=dalation_rate,
    #             padding='causal', activation="relu"
    #         )
    #     )
    # model.add(keras.layers.Conv1D(filters=1, kernel_size=1))
    #
    # lr_schedule = keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch / 30))
    # optimizer = keras.optimizers.Adam(lr=1e-4)
    # model.compile(
    #     loss=keras.losses.Huber(),
    #     optimizer=optimizer,
    #     metrics=["mae"]
    # )
    #
    # history = model.fit(
    #     train_set,
    #     epochs=100,
    #     callbacks=[lr_schedule]
    # )
    #
    # plt.semilogx(history.history["lr"], history.history["loss"])
    # plt.axis([1e-4, 1e-2, 0, 30])
    # plt.show()

    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=[None, 1]))
    for dalation_rate in (1, 2, 4, 8, 16, 32):
        model.add(
            keras.layers.Conv1D(
                filters=32, kernel_size=2,
                strides=1, dilation_rate=dalation_rate,
                padding='causal', activation="relu"
            )
        )
    model.add(keras.layers.Conv1D(filters=1, kernel_size=1))

    early_stopping = keras.callbacks.EarlyStopping(patience=50)
    model_checkpoint = keras.callbacks.ModelCheckpoint("saved_models\\best_checkpoint_RNNs_CNNs_pre_processing.h5",
                                                       save_best_only=True)
    optimizer = keras.optimizers.Adam(lr=3e-4)
    model.compile(
        loss=keras.losses.Huber(),
        optimizer=optimizer,
        metrics=["mae"]
    )

    history = model.fit(
        train_set,
        epochs=500,
        validation_data=valid_set,
        callbacks=[early_stopping, model_checkpoint]
    )

    model = keras.models.load_model('saved_models\\best_checkpoint_RNNs_CNNs_pre_processing.h5')

    rnn_forecast = model_forecast(model,
                                  series[..., np.newaxis],
                                  window_size=window_size)
    rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]
    plt.figure(figsize=(10, 6))
    plot_series(time_valid, x_valid)
    plot_series(time_valid, rnn_forecast)
    plt.show()

    print(keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy())