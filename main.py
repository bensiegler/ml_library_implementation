import numpy as np
import matplotlib.pyplot as plt


def predict_with_current_params(X, w, b):
    return np.dot(X, w) + b


def compute_cost(X, Y, w, b):
    Y_HAT = predict_with_current_params(X, w, b)
    diff_squared_sum = 0
    for i in range(len(Y_HAT)):
        diff_squared_sum += pow(Y_HAT[i] - Y[i], 2)
    return diff_squared_sum / (2 * len(Y_HAT))


def find_adjustments(X, Y, w, b):
    num_examples, num_input_features = X.shape
    w_adjustments = np.zeros((num_input_features,))
    b_adjustment = 0
    for i in range(num_examples):
        error = Y[i] - (np.dot(X[i], w) + b)
        for j in range(num_input_features):
            w_adjustments[j] = w_adjustments[j] + (error * X[i][j])
        b_adjustment += error
    return w_adjustments / num_examples, b_adjustment / num_examples


def adjust_weights_and_bias(X, Y, w, b, a):
    m, n = X.shape
    dw, db = find_adjustments(X, Y, w, b)
    for i in range(n):
        w[i] = w[i] + (dw[i] * a)
    b = b + (db * a)
    return w, b


def test_for_best_alpha(X, Y, testing_alphas, number_of_iters_per_test):
    for alpha_to_test, cost_tracker in testing_alphas.items():
        cost_tracker, _, __ = run_gradient_descent(X, Y, alpha_to_test, 0.001, number_of_iters_per_test)
        fig, ax = plt.subplots()
        plt.title(alpha_to_test)
        ax.scatter(range(number_of_iters_per_test), cost_tracker)
        plt.show()


def run_gradient_descent(X, Y, a, e, m):
    cost_tracker = []
    w = np.array(np.zeros(X[0].shape))
    b = 0
    for i in range(m):
        cost_tracker.append(compute_cost(X, Y, w, b))
        w, b = adjust_weights_and_bias(X, Y, w, b, a)
        if i > 1 and cost_tracker[i] - cost_tracker[i - 1] < e:
            break
    return cost_tracker, w, b


x = np.array([[1, 5],
              [2, 2],
              [3, 1],
              [4, 0]])  # input values
y = np.array([103,
              150,
              180,
              200])  # true values

test_for_best_alpha(x, y, {0.001: [], 0.01: [], 0.1: [], 1: []}, 10)
