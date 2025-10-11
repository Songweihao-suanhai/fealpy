import numpy as np

add_at_test_data = [
    # (array,indices,values,expected)
    # 1d
    (np.array([1, 2, 3, 4, 5]), np.array([0, 0, 1, 1, 1]), np.array([2, 4, 6, 8, 10]), np.array([7, 26, 3, 4, 5])), # 重复索引累加
    (np.array([1, 2, 3, 4]), np.array([1, 3, 2]), np.array([1, 3, 5]), np.array([1, 3, 8, 7])), # 单索引
    (np.array([1, 2, 3, 4]), np.array([], dtype=np.int64), np.array([]), np.array([1, 2, 3, 4])), # 空数组的数据类型为float型，而索引只支持整型或者bool型
    (np.array([2, 4, 6, 8]), np.array([-1, -2]), np.array([5, 10]), np.array([2, 4, 16, 13])), # 单个负索引
    (np.array([0, 0, 0, 0]), np.array([1, 3]), 2, np.array([0, 2, 0, 2])), # 广播
    # 2d元组类型
    (np.array([[1, 2], [3, 4]]), (np.array([0, 0]), np.array([1, 1])), np.array([2, 3]), np.array([[1, 7], [3, 4]])),
    (np.zeros((3, 3)), (np.array([0, 1, 2]), np.array([0, 1, 2])), np.array([1, 2, 3]), np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])),
    # 2d列表类型(把indices展平)
    (np.array([[1, 2], [3, 4]]), np.array([[0, 0]]), np.array([1, 2]), np.array([[3, 6], [3, 4]])),
    (np.zeros((3, 3)), np.array([[0, 1], [0, 0]]), np.array([1, 2, 3]), np.array([[3, 6, 9], [1, 2 ,3], [0, 0, 0]])),
    # 3d元组类型
    (np.zeros((2, 3, 3)), (np.array([1, 1, 0, 0]), 
                           np.array([0, 1, 2, 2]), 
                           np.array([1, 2, 1, 2])), np.array([2, 4 ,6, 8]), np.array([[[0, 0, 0], [0, 0, 0], [0, 6, 8]], 
                                                                                      [[0 ,2 ,0], [0, 0, 4], [0, 0, 0]]])), # 单索引
    (np.zeros((2, 2, 2)), (np.array([0, 1, 1]),
                           np.array([0, 1, 1]),
                           np.array([0, 1, 1])), np.array([1, 2, 3]), np.array([[[1, 0], [0, 0]], [[0, 0], [0, 5]]])), # 重复索引累加
    # 3d列表类型(把indices展平)
    (np.zeros((3, 3, 3)), np.array([[1, 1], [0, 2], [1, 1]]), np.array([[1, 2, 3],
                                                                        [2, 3, 4],
                                                                        [3, 4, 5]]), np.array([[[1, 2, 3], [2, 3, 4], [3, 4, 5]], 
                                                                                               [[4, 8, 12], [8, 12, 16], [12, 16, 20]],
                                                                                               [[1, 2, 3], [2, 3, 4], [3, 4, 5]]])),
]