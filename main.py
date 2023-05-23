x = np.array([[1, 5],
              [2, 2],
              [3, 1],
              [4, 0]])  # input values
y = np.array([103,
              150,
              180,
              200])  # true values

test_for_best_alpha(x, y, {0.001: [], 0.01: [], 0.1: [], 1: []}, 10)