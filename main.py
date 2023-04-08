import numpy as np
X = []
Y = []

w = []
b = 0

Y_HAT = []


def compute_cost(X, Y, w, b):
    J = 0
    for i in range(X):
        Y_HAT[i] = np.dot(X, w) + b
    



