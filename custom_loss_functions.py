import numpy as np

def huber_approx_obj(preds, dtrain):
    d = preds - dtrain
    h = 1
    scale = 1 + (d / h) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = -d / scale_sqrt
    hess = 1 / scale / scale_sqrt
    return grad, hess



def custom_loss_treshold_30(y_pred, y_true):
    grad = np.where(y_true <= 30, 2.5*(y_true-y_pred), 1)
    hess = np.where(y_true <= 30, 2.5, 0)
    return grad, hess

def custom_loss_treshold_50(y_pred, y_true):
    grad = np.where(y_true < 50, 2.5*(y_true-y_pred), 1)
    hess = np.where(y_true < 50, 2.5, 0)
    return grad, hess

def custom_loss_treshold_75(y_pred, y_true):
    grad = np.where(y_true < 75, 2.5*(y_true-y_pred), 1)
    hess = np.where(y_true < 75, 2.5, 0)
    return grad, hess

def custom_loss_test(y_pred, y_true):
    grad = np.where((y_true-y_pred) < 75, 2.5*(y_true-y_pred), 1)
    hess = np.where((y_true-y_pred) < 75, 2.5, 0)
    return grad, hess