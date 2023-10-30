import numpy as np
import matplotlib.pyplot as plt

class KMeansClustering:
    def __init__(self, k=3):
        """
        Initialize a KMeansClustering instance.

        :param k: The number of clusters (default is 3).
        """
        self.k = k
        self.centroids = None

    @staticmethod
    def euclidean_distance(data_points, centroids):
        """
        Calculate the Euclidean distance between data points and centroids.

        :param data_points: Array of data points.
        :param centroids: Array of centroids.
        :return: Array of distances between data points and centroids.
        """
        return np.sqrt(np.sum((centroids - data_points) ** 2, axis=1)

    def fit(self, X, max_iter=200):
        """
        Fit the K-Means clustering model to the input data.

        :param X: Input data for clustering.
        :param max_iter: Maximum number of iterations (default is 200).
        :return: Cluster labels for each data point.
        """
        # Initialize centroids as random data points
        self.centroids = X[np.random.choice(X.shape[0], size=self.k, replace=False)]

        for _ in range(max_iter):
            y = []

            # Assign each data point to the nearest centroid
            for data_point in X:
                distances = KMeansClustering.euclidean_distance(data_point, self.centroids)
                cluster_min = np.argmin(distances)
                y.append(cluster_min)

            y = np.array(y)

            # Calculate the new cluster centers
            cluster_idx = []
            for i in range(self.k):
                cluster_idx.append(np.argwhere(y == i))

            cluster_centers = []
            for i, idx in enumerate(cluster_idx):
                if len(idx) == 0:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X[idx], axis=0)[0])

            # Check for convergence
            if np.max(self.centroids - np.array(cluster_centers)) < 0.0001:
                break
            else:
                self.centroids = np.array(cluster_centers)

        return y

# Generate random data points for clustering
random_points = np.random.randint(0, 100, (100, 2))

# Create a KMeansClustering instance with 3 clusters
kmeans = KMeansClustering(k=3)

# Fit the clustering model to the random data points
labels = kmeans.fit(random_points)

# Plot the data points with cluster labels and centroids
plt.scatter(random_points[:, 0], random_points[:, 1], c=labels)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c=range(len(kmeans.centroids)), marker="*", s=200)
plt.show()
