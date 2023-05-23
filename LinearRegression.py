import numpy as np


def compute_cost(Y_HAT, Y):
    diff_squared_sum = 0
    for i in range(len(Y_HAT)):
        diff_squared_sum += pow(Y_HAT[i] - Y[i], 2)
    return diff_squared_sum / (2 * len(Y_HAT))


def predict_with_current_params(X, w, b):
    return np.dot(X, w) + b
