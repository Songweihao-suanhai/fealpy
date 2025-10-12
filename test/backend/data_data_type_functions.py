import numpy as np

astype_test_data = [
    # 格式：(array, dtype, expected)
    # ===== int 型 =====
    (np.array([2, 3, 4, 5, 6]), 'float64', np.array([2.0, 3.0, 4.0, 5.0, 6.0])),
    (np.array([2, -3, 0, 1, -5]), 'bool', np.array([True, True, False, True, True])),
    # ===== float 型 =====
    (np.array([1.0, 3.0, -5.0, 7.0, 9.0]), 'int64', np.array([1, 3, -5, 7, 9])),
    (np.array([-2.0, 3.0, 0.0, 1.0, -5.0]), 'bool', np.array([True, True, False, True, True])),
    (np.array([np.nan, np.inf]), 'bool', np.array([True, True])),
    # ===== bool 型 =====
    (np.array([True, True, False, True, True]), 'int64', np.array([1, 1, 0, 1, 1])),
    (np.array([True, True, False, True, True]), 'float64', np.array([1.0, 1.0, 0.0, 1.0, 1.0])),
    (np.array([True, False, True, False, True]), 'complex64', np.array([1+0j, 0+0j, 1+0j, 0+0j, 1+0j])),
    # ===== complex 型 =====
    (np.array([0+0j, 1+1j, 4+2j, -3+3j]), 'bool', np.array([False, True, True, True])),
    (np.array([1+2j, 3+4j, -5+6j]).real, 'int64', np.array([1, 3, -5])),
    (np.array([1+2j, 3+4j, -5+6j]).imag, 'float64', np.array([2.0, 4.0, 6.0])),
    # ===== 边界情况 =====
    (np.array([], dtype=int), 'float64', np.array([], dtype=float)),
    (np.array([], dtype=bool), 'int64', np.array([], dtype=int)),
    (np.array([], dtype=float), 'complex64', np.array([], dtype=complex)),
]