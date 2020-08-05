import  numpy as np
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


def seasonality(time, period, amplitude=1 , phase=0):
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
    return mov[window_size - 1 : -1] / window_size


def window_dataset(series, window_size, batch_size=32, shuffle_buffer=1000):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


if __name__ == '__main__':
    # time = np.arange(4 * 365 + 1)
    # baseline = 10
    # series = baseline + trend(time, 0.1)
    #
    # amplitude = 40
    #
    # slope = 0.05
    # series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
    #
    # noise_level = 5
    # noise = white_noise(time, noise_level, seed=42)
    #
    # series += noise
    # # plt.figure(figsize=(10, 6))
    # # plot_series(time, series)
    #
    # split_time = 1000
    # time_train = time[:split_time]
    # x_train = series[:split_time]
    # time_valid = time[split_time:]
    # x_valid = series[split_time:]

    dataset = tf.data.Dataset.range(10)
    # for val in dataset:
    #     print(val)

    dataset = dataset.window(5, shift=1, drop_remainder=True)
    # for window_dataset in dataset:
    #     for val in window_dataset:
    #         print(val.numpy(), end=" ")
    #     print()
    dataset = dataset.flat_map(lambda window: window.batch(5))
    # for val in dataset:
    #     print(val.numpy())
    dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
    # for x, y in dataset:
    #     print(x.numpy(), y.numpy())
    dataset = dataset.shuffle(buffer_size=10)
    # for x, y in dataset:
    #     print(x.numpy(), y.numpy())
    dataset = dataset.batch(2).prefetch(1)
    for x, y in dataset:
        print("x = ", x.numpy())
        print("y = ", y.numpy())