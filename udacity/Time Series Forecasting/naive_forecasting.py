import numpy as np
import matplotlib.pylab as plt


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

    naive_forecast = series[split_time - 1:-1]

    # plt.figure(figsize=(10, 6))
    # plot_series(time_valid, x_valid,start=0, end=150, label="series")
    # plot_series(time_valid, naive_forecast, start=1, end=151, label="Forecast")
    # plt.show()

    errors = naive_forecast - x_valid
    abs_errors = np.abs(errors)
    mae = abs_errors.mean()
    print(mae)