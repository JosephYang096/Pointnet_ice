import numpy as np

def random_select_points_from_txt(txt_file, num_points):
    """
    从 txt 文件中随机选择指定数量的点
    :param txt_file: 输入的 txt 文件路径
    :param num_points: 需要随机选择的点数
    :return: 随机选择的点的 numpy 数组
    """
    try:
        # 读取 txt 文件
        points = np.loadtxt(txt_file)
        total_points = points.shape[0]

        if total_points < num_points:
            print(f"文件中的点数 {total_points} 少于需要选择的点数 {num_points}，将返回所有点。")
            return points

        # 随机选择指定数量的点
        selected_indices = np.random.choice(total_points, num_points, replace=False)
        selected_points = points[selected_indices]

        return selected_points
    except FileNotFoundError:
        print(f"未找到文件: {txt_file}")
        return None
    except Exception as e:
        print(f"发生未知错误: {e}")
        return None

# 示例使用
txt_file = 'env.txt'  # 替换为你的 txt 文件路径
num_points = 10  # 需要随机选择的点数
selected_points = random_select_points_from_txt(txt_file, num_points)

if selected_points is not None:
    print("随机选择的点:")
    print(selected_points)