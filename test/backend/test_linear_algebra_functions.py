import pytest
from fealpy.backend import backend_manager as bm
from data_liner_algebra_functions import *

class TestLinearAlgebraFunctionsInterfaces:
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("a, b, axes, expected", tensordot_test_data)
    def test_tensordot(self, backend, a, b, axes, expected):
        '''
        Compute tensor dot product along specified axes.
        '''
        # 设置后端
        bm.set_backend(backend)
        # 转换为对应的后端数组
        a = bm.from_numpy(a)
        b = bm.from_numpy(b)
        expected = bm.from_numpy(expected)
        # 测试
        assert bm.all(bm.tensordot(a, b, axes) == expected)
        
if __name__ == '__main__':
    pytest.main(['test/backend/test_linear_algebra_functions.py', '-qs', '--disable-warnings'])