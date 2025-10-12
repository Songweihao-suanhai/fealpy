import numpy as np

tensordot_test_data = [
    # 格式：(a, b, axes, expected)
    # 1d
    (np.array([1, 2, 3]), np.array([4, 5, 6]), 0, np.array([[4, 5, 6], [8, 10, 12], [12, 15, 18]])), # 外积
    (np.array([1, 2, 3]), np.array([4, 5, 6]), 1, np.array([32])), # 内积
    # 2d(axes 为 int 型)
    (np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]), 0, np.array([[[[5, 6], [7, 8]],
                                                                          [[10, 12], [14, 16]]],
                                                                          [[[15, 18], [21, 24]],
                                                                          [[20, 24], [28, 32]]]])), 
    (np.array([[1, 3], [5, 7]]), np.array([[2, 4], [6, 8]]), 1, np.array([[20, 28], [52, 76]])),
    (np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]), 2, np.array([70])),
    # 2d(axes 为 tuple 型)
    (np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]), (0, 0), np.array([[26, 30], [38, 44]])), # TODO:pytorch不支持tuple类型
    (np.array([[1, 3], [5, 7]]), np.array([[2, 4], [6, 8]]), (1, 1), np.array([[14, 30], [38, 86]])),
    # 1d、2d 混合(axes 为 int 型)
    (np.array([1, 2, 3]), np.array([[2, 4], [1, 3]]), 0, np.array([[[2, 4], [1, 3]], 
                                                                   [[4, 8], [2, 6]], 
                                                                   [[6, 12], [3, 9]]])),
    (np.array([1, 2]), np.array([[1, 3], [5, 7]]), 1, np.array([11, 17])),
    # 1d、2d 混合(axes 为 tuple 型)
    (np.array([1, 2]), np.array([[1, 3], [5, 7]]), (0, 1), np.array([7, 19])),
]