import numpy as np

X = np.array([[1, 5],
              [2, 2],
              [3, 1],
              [4, 0]])  # input values
Y = np.array([103,
              150,
              180,
              200])  # true values

w = np.array(np.zeros(X[0].shape))  # feature weights
b = 0.0  # bias
a = 0.01
Y_HAT = []  # predicted values


def predict():
    global Y_HAT
    Y_HAT = np.dot(X, w) + b


def compute_cost():
    predict()
    diff_squared_sum = 0
    for i in range(len(Y_HAT)):
        diff_squared_sum += pow(Y_HAT[i] - Y[i], 2)
    return diff_squared_sum / (2 * len(Y_HAT))


def find_adjustments():
    num_examples, num_input_features = X.shape
    w_adjustments = np.zeros((num_input_features,))
    b_adjustment = 0
    for i in range(num_examples):
        error = Y[i] - (np.dot(X[i], w) + b)
        for j in range(num_input_features):
            w_adjustments[j] = w_adjustments[j] + (error * X[i][j])
        b_adjustment += error
    return w_adjustments / num_examples, b_adjustment / num_examples


for x in range(30000):
    print("cost", compute_cost())
    print("prediction", Y_HAT)
    m, n = X.shape
    dw, db = find_adjustments()
    print("w", w, "     dw", dw)
    print("b", b, "     db", db)
    for y in range(n):
        w[y] = w[y] + (dw[y] * a)
    b = b + (db * a)

