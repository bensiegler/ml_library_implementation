import numpy as np


def compute_cost(Y_HAT, Y):
    total = 0
    for i in range(len(Y_HAT)):
        total += (Y[i] * np.log(Y_HAT[i])) + (Y[i] - 1)(np.log(1 - Y_HAT[i]))

    return -total / len(Y_HAT)

def predict_with_current_params(X, w, b):
    Y_HAT = np.dot(X, w) + b
