import numpy as np


def custom_error(true_value, prediction):
    if true_value < prediction:  # Overestimation
        return np.exp(-(prediction - true_value)/10)
    else:  # Underestimation is less penalized
        return np.exp((prediction - true_value)/13)


def mean_custom_error(y_true, y_pred):
    return np.mean([custom_error(true, pred) for true, pred in zip(y_true, y_pred)])


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
