import numpy as np
import matplotlib.pyplot as plt

def extract_first_three_numbers(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            # 按空格分割每行内容
            numbers = line.strip().split()
            # 确保至少有三个数字
            if len(numbers) >= 3:
                try:
                    # 提取前三个数字并转换为浮点数
                    x, y, z = map(float, numbers[:3])
                    points.append([x, y, z])
                except ValueError:
                    print(f"无法将行 {line} 中的数字转换为浮点数。")
    return np.array(points)

def plot_scatter(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 提取 x, y, z 坐标
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    # 绘制散点图
    ax.scatter(x, y, z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

if __name__ == "__main__":
    file_path = 'env.txt'  # 替换为你的 TXT 文件路径
    points = extract_first_three_numbers(file_path)
    if points.size > 0:
        plot_scatter(points)
    else:
        print("未找到有效的点数据。")