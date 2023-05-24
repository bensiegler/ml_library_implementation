import numpy as np


def compute_cost(Y_HAT, Y):
    total = 0
    for i in range(len(Y_HAT)):
        total += (Y[i] * np.log(Y_HAT[i])) + (1 - Y[i])*(np.log(1 - Y_HAT[i]))
    return -total / len(Y_HAT)


def predict_with_current_params(X, w, b):
    Y_HAT = np.dot(X, w) + b
    for i in range(len(Y_HAT)):
        Y_HAT[i] = 1 / (1 + pow(np.e, -Y_HAT[i]))
    print(Y_HAT)
    return Y_HAT

