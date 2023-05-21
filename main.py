import numpy as np
import matplotlib.pyplot as plt

epsilon = 0.001
x = np.array([[1, 5],
              [2, 2],
              [3, 1],
              [4, 0]])  # input values
y = np.array([103,
              150,
              180,
              200])  # true values


# w = np.array(np.zeros(X[0].shape))  # feature weights
# b = 0.0  # bias

# a = 0.5


#
# Y_HAT = []  # predicted values


def predict_with_current_params(X, w, b):
    return np.dot (X, w) + b


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
        w = np.array(np.zeros(X[0].shape))
        b = 0
        for i in range(number_of_iters_per_test):
            cost_tracker.append(compute_cost(X, Y, w, b))
            w, b = adjust_weights_and_bias(X, Y, w, b, alpha_to_test)
        fig, ax = plt.subplots()
        plt.title(alpha_to_test)
        ax.scatter(range(number_of_iters_per_test), cost_tracker)
        plt.show()


test_for_best_alpha(x, y, {0.001: [], 0.01: [], 0.15: [], 1: []}, 20)

# costs = [compute_cost()]
# final_iteration = -1
# for curr_iteration in range(100000):
#
#
#     # automatic cost convergence check
#     costs.append(compute_cost())
#     dj_dx = costs[len(costs) - 2] - costs[len(costs) - 1]
#     if epsilon >= dj_dx > 0:
#         final_iteration = curr_iteration
#         fig, ax = plt.subplots()
#         ax.scatter(range(final_iteration + 2), costs)
#         plt.show()
#         break
#     elif costs[len(costs) - 2] - costs[len(costs) - 1] < -epsilon:
#         print("WARNING COST INCREASING MATERIALLY: trying again with alpha reduced by a factor of 2")
#         w = np.array(np.zeros(X[0].shape))  # feature weights
#         b = 0.0
#         costs = [compute_cost()]
#         a /= 2
