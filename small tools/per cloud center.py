import numpy as np
import os


def process_point_cloud(file_path):
    try:
        # 读取点云数据
        points = np.loadtxt(file_path)

        # 计算中心点坐标
        center = np.mean(points, axis=0)
        x, y, z = center

        # 按 z 坐标降序排序
        sorted_points = points[points[:, 2].argsort()[::-1]]

        # 取 z 最大的 20 个点
        top_20_points = sorted_points[:20]

        # 计算 z 最大的 20 个点的均值
        mean_top_20 = np.mean(top_20_points, axis=0)
        x2, y2, z2 = mean_top_20

        # 生成新文件名
        new_file_name = f"1_{x:.6f}_{y:.6f}_{z:.6f}_{x2:.6f}_{y2:.6f}_{z2:.6f}.txt"

        # 保存文件
        np.savetxt(new_file_name, points, fmt='%.6f', delimiter=' ')
        print(f"处理完成，新文件名为: {new_file_name}")

    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}。")
    except Exception as e:
        print(f"发生未知错误: {e}")


if __name__ == "__main__":
    file_path = "your_point_cloud.txt"  # 替换为你的 TXT 文件路径
    process_point_cloud(file_path)
