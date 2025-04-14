import numpy as np


def read_txt_with_column_range(file_path, column_range, row_range=None):
    """
    读取包含未知列数的 TXT 文件，并根据指定的列区间和行区间返回数据。

    参数:
    file_path (str): TXT 文件的路径
    column_range (tuple): 包含起始列和结束列的元组，例如 (1, 3) 表示读取第 1 列到第 3 列的数据
    row_range (tuple, 可选): 包含起始行和结束行的元组，例如 (0, 10) 表示读取第 0 行到第 10 行的数据，默认为 None，即读取全量行

    返回:
    np.ndarray: 包含指定行列区间数据的 numpy 数组
    """
    try:
        start_col, end_col = column_range
        data = np.loadtxt(file_path)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        if start_col < 0 or end_col >= data.shape[1] or start_col > end_col:
            raise ValueError("指定的列区间无效。")

        if row_range is not None:
            start_row, end_row = row_range
            if start_row < 0 or end_row >= data.shape[0] or start_row > end_row:
                raise ValueError("指定的行区间无效。")
            return data[start_row:end_row + 1, start_col:end_col + 1]
        else:
            return data[:, start_col:end_col + 1]

    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}。")
        return None
    except ValueError as e:
        print(f"错误：读取文件时发生值错误，{e}。")
        return None
    except Exception as e:
        print(f"错误：发生未知错误，{e}。")
        return None
