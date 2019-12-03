import pandas as pd
import numpy as np


def feature_engineer(train, test, sensors, functions, silent=True, drop_na=True):
    """
    Applies list of functions to sensor signals

    :param train: indexed training dataset by engine_id
    :param test: indexed testing dataset by engine_id
    :param sensors: frame columns to apply feature engineering to
    :param functions: Set of tuples of feature engineering functions and dictionary of parameters.
                      The functions take a pandas Series indexed by engine id as input
                      and outputs a transformed pandas Series
                      ex : {
                            (lag, {'n_lag': 10}),
                            (time_reversal_asymmetry, {'rolling_window_length': 20, 'n_lag': 5})
                            }
    :param silent: print along feature engineering evolution
    :param drop_na: drop rows containing NaN values after rolling window and shift operations
    :return: transformed training set and transformed testing set
    """
    train = train.copy()
    test = test.copy()
    if not silent:
        print('Applying feature engineering to training and testing sets')
    for frame in (train, test):
        for func, params in functions:
            if not silent:
                print('Applying {function} with {params}'.format(function=func.__name__, params=params))
            for sensor in sensors:
                new_column, column_name = func(signal=frame[sensor], **params)
                frame[column_name] = new_column
    if not drop_na:
        return train, test
    return train.dropna(), test.dropna()


def lag(signal, n_lag):
    """
    :param signal: sensor signal
    :param n_lag: lag to apply to the signal
    :return: Lagged signal
    """
    signal = signal.copy()
    for engine_id in set(signal.index):
        signal.loc[engine_id] = signal.loc[engine_id].shift(n_lag)

    name = '{signal}_lag_{n_lag}'.format(signal=signal.name, n_lag=n_lag)
    return signal, name


def rolling_mean(signal, time_window_length):
    """
    :param signal: sensor signal
    :param time_window_length: length of time window to compute rolling mean
    :return: Signal rolling mean
    """
    signal = signal.copy()
    for engine_id in set(signal.index):
        signal.loc[engine_id] = signal.loc[engine_id].rolling(time_window_length).mean()
    name = '{signal}_rolling_mean_{time_window_length}'.format(signal=signal.name,
                                                               time_window_length=time_window_length)
    return signal, name


def time_reversal_asymmetry(signal, n_lag, time_window_length):
    """
    :param signal: sensor signal
    :param n_lag: lag to take for computing time reversal asymmetry
    :param time_window_length: window length for time reversal asymmetry
    :return: Signal rolling time reversal asymmetry
    """
    def trasymmetry(x):
        n = len(x)
        x = np.asarray(x)
        if 2 * n_lag >= n:
            return 0
        else:
            one_lag = np.roll(x, -n_lag)
            two_lag = np.roll(x, 2 * -n_lag)
            return np.mean((two_lag * two_lag * one_lag - one_lag * x * x)[0:(n - 2 * n_lag)])

    signal = signal.copy()
    for engine_id in set(signal.index):
        signal.loc[engine_id] = signal.loc[engine_id].rolling(time_window_length).apply(lambda x: trasymmetry(x),
                                                                                        raw=False)

    name = '{signal}_time_reversal_asymmetry_window_{time_window_length}_lag_{n_lag}'\
        .format(signal=signal.name, time_window_length=time_window_length, n_lag=n_lag)
    return signal, name


def mean_derivative(signal, time_window_length):
    """
    :param signal:
    :param time_window_length:
    :return:
    """
    signal = signal.copy()
    for engine_id in set(signal.index):
        signal.loc[engine_id] = (signal.loc[engine_id] - signal.loc[engine_id].shift(1))\
            .rolling(time_window_length).mean()
    name = '{signal}_rolling_mean_derivative_{time_window_length}'.format(signal=signal.name,
                                                                          time_window_length=time_window_length)
    return signal, name


def rolling_min(signal, time_window_length):
    """
    :param signal: sensor signal
    :param time_window_length: length of time window to compute min
    :return: Signal rolling mean
    """
    signal = signal.copy()
    for engine_id in set(signal.index):
        signal.loc[engine_id] = signal.loc[engine_id].rolling(time_window_length).min()
    name = '{signal}_rolling_min_{time_window_length}'.format(signal=signal.name,
                                                              time_window_length=time_window_length)
    return signal, name


def rolling_max(signal, time_window_length):
    """
    :param signal: sensor signal
    :param time_window_length: length of time window to compute max
    :return: Signal rolling mean
    """
    signal = signal.copy()
    for engine_id in set(signal.index):
        signal.loc[engine_id] = signal.loc[engine_id].rolling(time_window_length).max()
    name = '{signal}_rolling_max_{time_window_length}'.format(signal=signal.name,
                                                              time_window_length=time_window_length)
    return signal, name


def rolling_variance(signal, time_window_length):
    """
    :param signal: sensor signal
    :param time_window_length: length of time window to compute variance
    :return: Signal rolling mean
    """
    signal = signal.copy()
    for engine_id in set(signal.index):
        signal.loc[engine_id] = signal.loc[engine_id].rolling(time_window_length).max()
    name = '{signal}_rolling_variance_{time_window_length}'.format(signal=signal.name,
                                                                   time_window_length=time_window_length)
    return signal, name


def rolling_abs_energy(signal, time_window_length):
    """
    :param signal: sensor signal
    :param time_window_length: length of time window to compute max
    :return: Signal rolling mean
    """
    signal = signal.copy()
    for engine_id in set(signal.index):
        signal.loc[engine_id] = signal.loc[engine_id].apply(lambda x: x**2).rolling(time_window_length).sum()
    name = '{signal}_rolling_energy_{time_window_length}'.format(signal=signal.name,
                                                                 time_window_length=time_window_length)
    return signal, name


def rolling_abs_sum_of_changes(signal, time_window_length):
    """
    :param signal: sensor signal
    :param time_window_length: length of time window to compute max
    :return: Signal rolling mean
    """
    signal = signal.copy()
    for engine_id in set(signal.index):
        signal.loc[engine_id] = np.abs(signal.loc[engine_id] - signal.loc[engine_id].shift(1))
        signal.loc[engine_id] = signal.loc[engine_id].rolling(time_window_length).sum()
    name = '{signal}_rolling_sum_of_changes_{time_window_length}'.format(signal=signal.name,
                                                                         time_window_length=time_window_length)
    return signal, name



if __name__ == '__main__':
    df = pd.read_csv("CMAPSSData/CMAPSSData/train_FD001.txt", delimiter=" ", index_col=None, header=None)
    df.columns = ["col{}".format(i) for i in range(df.shape[1])]
    df = df.set_index('col0').loc[:10, :]
    test_signal = df['col6']

    print('Running unit tests for feature engineering functions')
    feature_engineering_functions = [
        (lag, {'n_lag': 10}),
        (rolling_mean, {'time_window_length': 20}),
        (mean_derivative, {'time_window_length': 20}),
        (rolling_max, {'time_window_length': 20}),
        (rolling_min, {'time_window_length': 20}),
        (rolling_abs_energy, {'time_window_length': 20}),
        (rolling_abs_sum_of_changes, {'time_window_length': 20}),
        (rolling_variance, {'time_window_length': 20}),
        (time_reversal_asymmetry, {'time_window_length': 10, 'n_lag': 2})
    ]
    failed_functions = []
    for func, params in feature_engineering_functions:
        try :
            col, name = func(test_signal, **params)
        except Exception as e:
            print('FAILED {}'.format(func.__name__))
            failed_functions.append(func.__name__)
    if not failed_functions:
        print('All function passed the test')
    else:
        print('{} functions failed : {}'.format(len(failed_functions), failed_functions))

