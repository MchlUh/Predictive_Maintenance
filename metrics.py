import numpy as np


def custom_error_metric(true_value, prediction):
    if true_value < prediction:  # Overestimation
        return np.exp(-(prediction - true_value)/10)
    else:  # Underestimation is less penalized
        return np.exp((prediction - true_value)/13)


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100