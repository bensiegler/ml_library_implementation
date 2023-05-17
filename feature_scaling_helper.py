import numpy as np


def perform_max_normalization(feature_values):
    maximum = max(feature_values)
    for i in range(len(feature_values)):
        feature_values[i] /= maximum


def perform_z_score_normalization(feature_values):
    sd = np.pstdev(feature_values)
    mean = np.mean(feature_values)
    for i in range(len(feature_values)):
        feature_values[i] = (feature_values[i] - mean) / sd


def perform_mean_normalization(feature_values):
    mean = np.mean(feature_values)
    spread = np.max(feature_values) - np.min(feature_values)
    for i in range(len(feature_values)):
        feature_values[i] = (feature_values[i] - mean) / spread

