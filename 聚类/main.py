import numpy as np
from kd_tree_clustering_module import cluster_and_visualize
from txt_point_cloud_reader import read_txt_with_column_range

#读取点云
file_path = '1.txt'
# 指定列区间，例如读取第 0 列到第 2 列
column_range = (0, 2)
# 指定行区间，例如读取第 1 行到第 5 行
row_range = (1, 5000)
points = read_txt_with_column_range(file_path, column_range, row_range)


#展示点云实例以及形状
if points is not None:
    print("读取的数据形状:", points.shape)
    print("部分数据示例:", points[:5])

# 调用函数进行聚类和可视化（可选）
centroids = cluster_and_visualize(points, eps=0.1, min_samples=10, visualize=True)

#展示点云标签与质心
print("聚类标签及对应质心:")
for label, centroid in centroids:
    print(f"Label: {label}, Centroid: {centroid}")

