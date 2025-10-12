import numpy as np

repeat_test_data = [
    # 格式：(array, repeat_counts, axis, expected)
    # ===== 1D =====
    (np.array([]), 2, None, np.array([])),  # 空数组重复，结果仍为空
    (np.array([1.0, 2.0, 3.0]), 1, None, np.array([1.0, 2.0, 3.0])),  # repeat=1，结果等于原数组
    (np.array([1, 2, 3]), np.array([1, 2, 3]), None, np.array([1, 2, 2, 3, 3, 3])),  # 每个元素不同重复次数
    (np.array([1, 2, 3]), 0, None, np.array([])),  # repeat=0，结果为空
    # ===== 2D =====
    (np.array([[]]), 2, 1, np.array([[]])),  # 空二维数组，沿 axis=1 重复，结果仍为空
    (np.array([[1, 2], [3, 4]]), 2, 0, np.array([[1, 2], [1, 2], [3, 4], [3, 4]])),  # 沿 axis=0 重复行
    (np.array([[1, 2], [3, 4]]), 2, 1, np.array([[1, 1, 2, 2], [3, 3, 4, 4]])),  # 沿 axis=1 重复列
    (np.array([[True, True], [False, True]]), 3, None,
     np.array([True, True, True, True, True, True, False, False, False, True, True, True])),  # axis=None，展平后整体重复
    (np.array([[1, 2], [3, 4]]), 0, 0, np.empty((0, 2))),  # 沿 axis=0，repeat=0，结果 shape=(0,2)
    (np.array([[1, 2], [3, 4]]), 0, 1, np.empty((2, 0))),  # 沿 axis=1，repeat=0，结果 shape=(2,0)
    (np.array([[1, 2], [3, 4]]), np.array([1, 2]), 0, np.array([[1, 2], [3, 4], [3, 4]])),  # 每行不同重复次数
    (np.array([[1, 2], [3, 4]]), np.array([2, 1]), 1, np.array([[1, 1, 2], [3, 3, 4]])),  # 每列不同重复次数
    # ===== 3D =====
    (np.array([[[]]]), 2, 2, np.array([[[]]])),  # 空三维数组，沿 axis=2 重复，结果仍为空
    (np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), 2, 0,
     np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]], [[5, 6], [7, 8]], [[5, 6], [7, 8]]])),  # 沿 axis=0 重复 block
    (np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), 2, 1,
     np.array([[[1, 2], [1, 2], [3, 4], [3, 4]], [[5, 6], [5, 6], [7, 8], [7, 8]]])),  # 沿 axis=1 重复行
    (np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), 2, 2,
     np.array([[[1, 1, 2, 2], [3, 3, 4, 4]], [[5, 5, 6, 6], [7, 7, 8, 8]]])),  # 沿 axis=2 重复列
    (np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), 4, None,
     np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
               5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8])),  # axis=None，展平后整体重复
    (np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), 0, 0, np.empty((0, 2, 2))),  # 沿 axis=0，repeat=0
    (np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), 0, 1, np.empty((2, 0, 2))),  # 沿 axis=1，repeat=0
    (np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), 0, 2, np.empty((2, 2, 0))),  # 沿 axis=2，repeat=0
]