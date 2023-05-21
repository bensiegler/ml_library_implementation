import numpy as np


def max_norm(value, maximum): value / maximum


def z_score_norm(value, mean, sd): (value - mean) / sd


def mean_norm(value, mean, spread): (value - mean) / spread


def perform_max_normalization(feature_values):
    maximum = max(feature_values)
    return map(max_norm, feature_values, maximum)


def perform_z_score_normalization(feature_values):
    sd = np.pstdev(feature_values)
    mean = np.mean(feature_values)
    return map(z_score_norm, feature_values, sd, mean)


def perform_mean_normalization(feature_values):
    mean = np.mean(feature_values)
    spread = np.max(feature_values) - np.min(feature_values)
    return map(mean_norm, feature_values, mean, spread)
