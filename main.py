import numpy as np

X = [[100, 50, 25]]  # input values [[1, 2, 3]]
Y = [1, 2, 4]  # true values

w = [0]  # feature weights
b = [0]  # bias

Y_HAT = []  # predicted values


def predict_values():
    for i in range(len(X)):
        Y_HAT.append(np.dot(X[i], w[i]) + b[i])
    return Y_HAT


def compute_cost():
    predict_values()
    diff_squared_sum = 0
    for i in range(len(Y_HAT)):
        diff_squared_sum += pow(Y_HAT[i] - Y[i], 2)
    return diff_squared_sum / 2 * len(Y_HAT)


def find_w_adjustments():
    num_input_features = len(X[0])
    num_examples = len(X)
    w_adjustments = []
    for i in range(num_input_features):
        for j in range(num_examples):
            w_adjustments.append(((Y_HAT[i] - Y[i]) * X[i][j]) / num_examples)
    return w_adjustments


print(compute_cost())
print(find_w_adjustments())
