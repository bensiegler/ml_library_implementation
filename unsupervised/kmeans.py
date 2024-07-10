import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class KMeans:

    def __init__(self, vectors, k, iterations, num_initializations, epsilon):
        self.epsilon = epsilon
        self.best_centroids = None
        self.iterations = iterations
        self.k = k
        self.num_initializations = num_initializations;
        self.vectors = vectors
        self.centroids = np.zeros((k, self.vectors.shape[1]))
        self.c = np.zeros(len(self.vectors), dtype=int)
        if len(self.vectors) < k:
            raise ValueError("The number of vector cannot be less than the number of centroids.")

    def run(self):
        best_cost = float('inf')
        last_cost = float('inf')
        for initialization in range(self.num_initializations):
            self.initialize_centroids()
            for i in range(self.iterations):
                self.find_closest_centroid_to_vector()
                last_cost = self.compute_cost()
                self.move_centroids_to_avg()

            self.find_closest_centroid_to_vector()
            current_cost = self.compute_cost()
            if  current_cost < best_cost:
                best_cost = current_cost
                self.best_centroids = self.centroids.__copy__()
            if np.abs(last_cost - current_cost) <= self.epsilon:
                break

    def initialize_centroids(self):
        chosen_indices = np.random.choice(len(self.vectors), self.k, replace=False)
        self.centroids = self.vectors[chosen_indices]

    def compute_cost(self):
        cost = 0
        for vector_index in range(len(self.vectors)):
            best_centroid = self.c[vector_index]
            cost += np.sum(np.square(self.centroids[best_centroid] - self.vectors[vector_index]))
        return cost / len(self.vectors)

    def move_centroids_to_avg(self):
        for u in range(len(self.centroids)):
            assigned_vectors = self.vectors[self.c == u]
            if len(assigned_vectors) > 0:
                self.centroids[u] = np.mean(assigned_vectors, axis=0)

    def find_closest_centroid_to_vector(self):
        for vector_index in range(len(self.vectors)):
            distances = np.linalg.norm(self.centroids - self.vectors[vector_index], axis=1)
            self.c[vector_index] = np.argmin(distances)
        for u in range(len(self.centroids)):
            if np.sum(self.c == u) == 0:
                self.initialize_centroids()
                self.find_closest_centroid_to_vector()

    def find_distance(self, centroid, vector):
        current_distance = 0
        for feature_index in range(vector.shape[0]):
            current_distance += np.square(centroid[feature_index] - vector[feature_index])
        return np.sqrt(current_distance)