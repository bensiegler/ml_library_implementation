import numpy as np

X = [[1, 2, 3]]  # input values
Y = [1, 2, 4]  # true values

w = [0]  # feature weights
b = [0]  # bias
a = 0.01
Y_HAT = []  # predicted values


def predict_values():
    global Y_HAT
    # iterate through each example and compute the predicted value
    for i in range(len(X)):
        Y_HAT = np.dot(X[i], w[i]) + b[i]
    return Y_HAT


def compute_cost():
    predict_values()
    diff_squared_sum = 0
    for i in range(len(Y_HAT)):
        diff_squared_sum += pow(Y_HAT[i] - Y[i], 2)
    return diff_squared_sum / 2 * len(Y_HAT)


def find_w_adjustments():
    num_input_features = len(X)
    num_examples = len(X[0])
    w_adjustments = []
    total = 0
    for i in range(num_input_features):
        for j in range(num_examples):
            total += ((Y_HAT[i] - Y[i]) * X[i][j]) / num_examples
        w_adjustments.append(total)
    return w_adjustments


for j in range(5000):
    print("cost", compute_cost())
    print("prediction", Y_HAT)

    dw = find_w_adjustments()
    print("w", w, "     dw", dw)
    for i in range(len(dw)):
        w[i] = w[i] - (dw[i] * a)

