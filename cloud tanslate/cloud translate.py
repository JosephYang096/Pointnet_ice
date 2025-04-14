import os
import numpy as np
import random


# 读取env.txt中的点云数据
def read_env_file(env_path):
    with open(env_path, 'r') as file:
        env_points = [line.strip().split() for line in file.readlines()]
    # 将点云数据转为float
    env_points = np.array(env_points, dtype=np.float32)
    return env_points


# 解析冰锥点云文件并返回数据
def read_cone_file(cone_file_path):
    # 从文件名中提取信息：文件名格式为1_a_b_c_d_e_f
    filename = os.path.basename(cone_file_path)
    parts = filename.split('_')

    # 文件编号为parts[0]，质量中心的XYZ为a, b, c，顶点XYZ为d, e, f
    mass_center = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
    apex_point = np.array([float(parts[4]), float(parts[5]), float(parts[6])])

    # 读取点云数据
    cone_data = np.loadtxt(cone_file_path)

    return mass_center, apex_point, cone_data


# 平移冰锥点云
def translate_cone(cone_data, apex_point, target_position):
    # 计算平移向量
    translation_vector = target_position - apex_point
    # 平移点云
    translated_cone = cone_data + translation_vector
    return translated_cone


# 随机选择k个冰锥文件并平移到目标位置
def process_cones_and_translate(cones_folder_path, k, target_positions):
    # 获取所有冰锥文件路径
    cone_files = [os.path.join(cones_folder_path, f) for f in os.listdir(cones_folder_path) if f.endswith('.txt')]

    # 随机选择k个冰锥文件
    selected_files = random.sample(cone_files, k)

    translated_cones = []

    # 读取每个冰锥文件并平移
    for i, cone_file in enumerate(selected_files):
        # 读取冰锥数据
        mass_center, apex_point, cone_data = read_cone_file(cone_file)

        # 获取目标位置
        target_position = np.array(target_positions[i])

        # 平移冰锥点云
        translated_cone = translate_cone(cone_data, apex_point, target_position)

        # 将平移后的点云添加到结果中
        translated_cones.append(translated_cone)

    # 合并所有平移后的冰锥点云
    all_translated_cones = np.vstack(translated_cones)

    return all_translated_cones


# 将平移后的冰锥点云追加到env.txt中
def append_to_env_file(env_path, translated_cones):
    with open(env_path, 'a') as file:
        for cone in translated_cones:
            for point in cone:
                file.write(' '.join(map(str, point)) + '\n')


# 主函数
def main(env_file, cones_folder, k, target_positions_file):
    # 1. 读取env.txt中的点云
    env_points = read_env_file(env_file)

    # 2. 读取目标位置文件
    target_positions = []
    with open(target_positions_file, 'r') as file:
        file.readline()  # 跳过第一行数字k
        target_positions = [list(map(float, line.strip().split())) for line in file.readlines()]

    # 3. 处理冰锥文件夹，平移冰锥点云
    translated_cones = process_cones_and_translate(cones_folder, k, target_positions)

    # 4. 将平移后的冰锥点云追加到env.txt
    append_to_env_file(env_file, translated_cones)

    print(f"Successfully added {k} translated cones to {env_file}.")


# 用户输入的路径和数字
env_file_path = 'path_to_your_env.txt'  # 这里替换为实际的env.txt路径
cones_folder_path = 'path_to_your_cone_folder'  # 这里替换为实际的冰锥点云文件夹路径
target_positions_file = 'path_to_target_positions.txt'  # 这里替换为实际的目标位置文件路径

# 读取目标冰锥个数
with open(target_positions_file, 'r') as f:
    k = int(f.readline().strip())  # 读取第一行数字k

# 执行主程序
main(env_file_path, cones_folder_path, k, target_positions_file)
