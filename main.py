import numpy as np
from Model import Model
import LinearRegression
import LogisticRegression

x = np.array([[1, 1],
              [0, 1],
              [1, 1],
              [0, 0]])  # input values
y = np.array([1,
              0,
              1,
              1])  # true values
X_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])
m = Model(X_train, y_train, 0.0001, LogisticRegression)
# m.test_for_best_alpha({0.001: [], 0.01: [], 0.1: [], 1: []}, 1000)
m.run_gradient_descent(0.1, 0.000001, 30000)
print(m.w, m.b)
