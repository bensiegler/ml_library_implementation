import numpy as np
import matplotlib.pyplot as plt


class Model:

    def __init__(self, X, Y, e, regression_type):
        self.X = X
        self.Y = Y
        self.e = e
        self.w = np.zeros(X[0].shape)
        self.b = 0
        self.regression_type = regression_type

    def find_adjustments(self, Y_HAT):
        num_examples, num_input_features = self.X.shape
        w_adjustments = np.zeros(num_input_features)
        b_adjustment = 0
        for i in range(num_examples):
            error = self.Y[i] - Y_HAT[i]
            for j in range(num_input_features):
                w_adjustments[j] = w_adjustments[j] + (error * self.X[i][j])
            b_adjustment += error
        return w_adjustments / num_examples, b_adjustment / num_examples

    def adjust_weights_and_bias(self, a, Y_HAT):
        m, n = self.X.shape
        dw, db = self.find_adjustments(Y_HAT)
        for i in range(n):
            self.w[i] = self.w[i] + (dw[i] * a)
        self.b = self.b + (db * a)

    def test_for_best_alpha(self, testing_alphas, number_of_iters_per_test):
        for alpha_to_test, cost_tracker in testing_alphas.items():
            cost_tracker, _, __ = self.run_gradient_descent(alpha_to_test, 0.001, number_of_iters_per_test)
            fig, ax = plt.subplots()
            plt.title(alpha_to_test)
            ax.scatter(range(len(cost_tracker)), cost_tracker)
            plt.show()
            self.w = np.zeros(self.X[0].shape)
            self.b = 0

    def run_gradient_descent(self, a, e, m):
        cost_tracker = []
        for i in range(m):
            prediction = self.regression_type.predict_with_current_params(self.X, self.w, self.b)
            curr_cost = self.regression_type.compute_cost(prediction, self.Y)
            cost_tracker.append(curr_cost)
            self.adjust_weights_and_bias(a, prediction)
            if i > 1 and (cost_tracker[i - 1] - cost_tracker[i]) < e:
                break
        return cost_tracker, self.w, self.b
