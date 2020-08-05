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

    # moving average
    moving_avg = moving_average_forecast_fast(series, 30)[split_time - 30:]
    print(keras.metrics.mean_absolute_error(x_valid, moving_avg).numpy())  # 7.14
    # plt.figure()
    # plot_series(time_valid, x_valid, label="Series")
    # plot_series(time_valid, moving_avg, label="Moving average (30 days)")
    # plt.show()

    # differencing
    diff_series = (series[365:] - series[:-365])
    diff_time = time[365:]

    # plt.figure(figsize=(10, 6))
    # plot_series(diff_time, diff_series, label="Series(t) - Series(t - 365)")
    # plt.show()

    diff_moving_avg = moving_average_forecast_fast(diff_series, 50)[split_time - 365 - 50:]
    # plt.figure()
    # plot_series(time_valid, diff_series[split_time - 365:], label="Series")
    # plot_series(time_valid, diff_moving_avg, label="Moving average (30 days)")
    # plt.show()

    diff_moving_avg_plus_past = series[split_time - 365:-365] + diff_moving_avg
    # plt.figure()
    # plot_series(time_valid, x_valid, label="Series")
    # plot_series(time_valid, diff_moving_avg_plus_past, label="Moving average (30 days)")
    # plt.show()

    diff_moving_avg_plus_smooth_past = moving_average_forecast_fast(series[split_time - 370:-359], 11) + diff_moving_avg
    print(keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())
    plt.figure()
    plot_series(time_valid, x_valid, label="Series")
    plot_series(time_valid, diff_moving_avg_plus_smooth_past, label="Moving average (30 days)")
    plt.show()