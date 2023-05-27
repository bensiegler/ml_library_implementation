import numpy as np
from Node import Node
from activation_functions import LogisticActivation

x = np.array([[1.2, 2.1, 3.5],
              [2.1, 1.3, 4.7],
              [3.0, 2.5, 1.8],
              [2.5, 3.2, 2.9],
              [4.2, 3.1, 2.5],
              [1.6, 2.8, 1.2],
              [2.9, 1.5, 4.2],
              [3.8, 2.7, 3.9],
              [1.7, 3.5, 2.1],
              [3.2, 1.9, 4.5],
              [2.4, 3.0, 1.7],
              [2.0, 1.4, 2.6],
              [3.5, 2.2, 3.0],
              [1.9, 3.3, 2.8],
              [2.7, 1.8, 1.5],
              [3.4, 2.6, 3.8]])  # input values
y = np.array([0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1])  # true values
X_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])
m = Node(x, y, 0.0001, LogisticActivation)
# m.test_for_best_alpha({0.001: [], 0.01: [], 0.1: [], 1: []}, 1000)
m.run_gradient_descent(0.1, 0.000001, 30000)
print(m.w, m.b)
