import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def dbscan_kdtree(points, eps, min_samples):
    tree = KDTree(points)
    n = len(points)
    labels = np.full(n, -1, dtype=int)
    cluster_id = 0

    for i in range(n):
        if labels[i] != -1:
            continue
        neighbors = tree.query_ball_point(points[i], eps)

        if len(neighbors) < min_samples:
            labels[i] = -1
            continue

        cluster_id += 1
        labels[i] = cluster_id
        seed_set = set(neighbors)
        seed_set.remove(i)

        while seed_set:
            j = seed_set.pop()
            if labels[j] == -1:
                labels[j] = cluster_id
                new_neighbors = tree.query_ball_point(points[j], eps)
                if len(new_neighbors) >= min_samples:
                    for neighbor in new_neighbors:
                        if labels[neighbor] == -1:
                            seed_set.add(neighbor)

    return labels


def cluster_and_visualize(points, eps=1.5, min_samples=5, visualize=True):
    labels = dbscan_kdtree(points, eps, min_samples)

    unique_labels = np.unique(labels)
    centroids = []
    for label in unique_labels:
        cluster_points = points[labels == label]
        if len(cluster_points) > 0:
            centroid = np.mean(cluster_points, axis=0)
            centroids.append([label, centroid])

    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            if label == -1:
                color = 'k'
            cluster_points = points[labels == label]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], c=[color], s=10)

            if label != -1:
                centroid = np.mean(cluster_points, axis=0)
                ax.scatter(centroid[0], centroid[1], centroid[2], c='r', s=50, marker='x')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('Point Cloud Clustering using DBSCAN with KD - Tree')
        plt.show()

    return centroids

