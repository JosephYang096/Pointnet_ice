import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_and_visualize_txt(file_path):
    # 读取txt文件，假设文件中的数据是空格分割的
    data = np.loadtxt(file_path)

    # 提取前三列作为XYZ坐标
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    # 创建3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制3D散点图
    ax.scatter(x, y, z, c='b', marker='o', s=10)

    # 设置标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 设置相同的比例尺
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()  # 计算三个轴的最大范围
    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5

    # 设置轴的范围，使得三个轴的范围相同
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

    # 显示图形
    plt.show()



# 使用示例
file_path = 'ice_point_cloud/2_-1.62_-0.98_0.03_-1.61_-0.99_0.10.txt'
load_and_visualize_txt(file_path)
