import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class KMeans:

    def __init__(self, vectors, k, iterations, num_initializations):
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
        for initialization in range(self.num_initializations):
            for i in range(self.iterations):
                self.initialize_centroids()
                self.find_closest_centroid_to_vector()
                self.move_centroids_to_avg()

            self.find_closest_centroid_to_vector()
            current_cost = self.compute_cost()
            if  current_cost < best_cost:
                best_cost = current_cost
                self.best_centroids = self.centroids

    def initialize_centroids(self):
        chosen_indices = np.random.choice(len(self.vectors), self.k, replace=False)
        self.centroids = self.vectors[chosen_indices]

    def compute_cost(self):
        cost = 0
        for vector in range(len(self.vectors)):
            best_centroid = self.c[vector]
            cost += np.square(self.find_distance(self.centroids[best_centroid], self.vectors[vector]))
        return cost / len(self.vectors)

    def move_centroids_to_avg(self):
        for u in range(len(self.centroids)):
            avg = np.zeros(self.vectors.shape[1])
            for i in range(len(self.c)):
                if self.c[i] == u:
                    avg += self.vectors[i]
            avg /= np.count_nonzero(self.c == u)
            self.centroids[u] = avg

    def find_closest_centroid_to_vector(self):
        for vector in range(len(self.vectors)):
            closest_distance = float('inf')
            best_centroid = -1
            for u in range(len(self.centroids)):
                current_distance = self.find_distance(self.centroids[u], self.vectors[vector])
                if current_distance < closest_distance:
                    closest_distance = current_distance
                    best_centroid = u
            self.c[vector] = best_centroid

    def find_distance(self, centroid, vector):
        current_distance = 0
        for i in range(vector.shape[0]):
            current_distance += np.square(centroid[i] - vector[i])
        return np.sqrt(current_distance)