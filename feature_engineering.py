import pandas as pd
import numpy as np


def feature_engineer(train, test, sensors, functions, silent=True):
    """
    :param train: indexed training dataset by engine_id
    :param test: indexed testing dataset by engine_id
    :param sensors: frame columns to apply feature engineering to
    :param functions: Dict of feature engineering functions as keys
                      with dictionary of parameters as value.
                      The functions take a pandas Series indexed by engine id as input
                      and outputs a transformed pandas Series
                      ex : {
                            lag: {'n_lag': 10},
                            time_reversal_asymmetry: {'rolling_window_length': 20, 'n_lag': 5}
                            }
    :return: transformed training set and transformed testing set
    """
    train = train.copy()
    test = test.copy()
    for frame in (train, test):
        for func, params in functions.items():
            if not silent:
                print('Applying {}...'.format(func.__name__))
            for sensor in sensors:
                new_column, column_name = func(signal=frame[sensor], **params)
                frame[column_name] = new_column
    return train, test


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
    return


def sofyas_super_genius_feature_engineering_function(pussy, ass, dick):
    return pussy, ass == dick
