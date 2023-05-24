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
m = Model(x, y, 0.0000001, LogisticRegression)
m.test_for_best_alpha({0.001: [], 0.01: [], 0.1: [], 1: []}, 1000)
# m.run_gradient_descent(0.01, 0.000001, 100000)
