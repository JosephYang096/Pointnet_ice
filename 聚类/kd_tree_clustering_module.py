import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


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

def calculate_mean_of_top_z_points(cluster_points):
    # 计算 z 最大的 50 个点的均值
    if len(cluster_points) > 50:
        top_z_points = cluster_points[np.argsort(cluster_points[:, 2])[-50:]]
    else:
        top_z_points = cluster_points
    return np.mean(top_z_points, axis=0)

def cluster_and_visualize(points, eps=1.5, min_samples=5, visualize=True, file=False):
    labels = dbscan_kdtree(points, eps, min_samples)

    unique_labels = np.unique(labels)
    centroids = []

    if file:
        # 创建文件夹
        os.makedirs('ice_point_cloud', exist_ok=True)

    for label in unique_labels:
        cluster_points = points[labels == label]
        if len(cluster_points) > 0:
            centroid = np.mean(cluster_points, axis=0)
            centroids.append([label, centroid])

            if file:
                # 计算 z 最大的 50 个点的均值
                mean_top_z = calculate_mean_of_top_z_points(cluster_points)
                # 生成文件名
                filename = f'ice_point_cloud/{label}_{centroid[0]:.2f}_{centroid[1]:.2f}_{centroid[2]:.2f}_{mean_top_z[0]:.2f}_{mean_top_z[1]:.2f}_{mean_top_z[2]:.2f}.txt'
                # 保存点云数据
                np.savetxt(filename, cluster_points, delimiter=' ', fmt='%.6f')

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
        plt.show()

    return centroids

