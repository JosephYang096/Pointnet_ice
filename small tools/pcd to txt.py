import numpy as np

def pcd_to_txt(pcd_file, txt_file):
    # 读取PCD文件的头部信息，确定数据格式
    with open(pcd_file, 'r', errors='ignore') as f:
        lines = f.readlines()
        data_format = None
        for line in lines:
            if line.startswith('DATA'):
                data_format = line.split()[1].strip()
                break

    if data_format == 'ascii':
        # ASCII格式的PCD文件
        start_index = 0
        for i, line in enumerate(lines):
            if line.startswith('DATA ascii'):
                start_index = i + 1
                break

        points = []
        for line in lines[start_index:]:
            parts = line.strip().split()
            points.append([float(part) for part in parts])

        points = np.array(points)
    elif data_format == 'binary':
        # 二进制格式的PCD文件
        import struct
        # 找到POINTS和DATA行
        points_num = None
        for line in lines:
            if line.startswith('POINTS'):
                points_num = int(line.split()[1].strip())
                break

        if points_num is None:
            raise ValueError("无法找到POINTS信息")

        # 找到数据起始位置
        data_start = None
        for i, line in enumerate(lines):
            if line.startswith('DATA binary'):
                data_start = i + 1
                break

        if data_start is None:
            raise ValueError("无法找到DATA信息")

        # 读取二进制数据
        with open(pcd_file, 'rb') as f:
            # 跳过头部信息
            for _ in range(data_start):
                f.readline()

            points = []
            for _ in range(points_num):
                # 假设每个点有4个浮点数（x, y, z, label）
                point_data = f.read(4 * 4)
                x, y, z, label = struct.unpack('ffff', point_data)
                points.append([x, y, z, label])

        points = np.array(points)
    else:
        raise ValueError(f"不支持的PCD数据格式: {data_format}")

    # 将点云数据保存为TXT文件
    np.savetxt(txt_file, points, fmt='%.6f', delimiter=' ')

# 示例使用
pcd_file = 'input.pcd'  # 输入的PCD文件
txt_file = 'output.txt'  # 输出的TXT文件

pcd_to_txt(pcd_file, txt_file)
print(f"转换完成，保存为 {txt_file}")