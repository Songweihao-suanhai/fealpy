import pytest
from fealpy.backend import backend_manager as bm
from data_non_standard import *

class TestNonStandardInterfaces:
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("array,indicies,values,expected", add_at_test_data)
    def test_add_at(self, backend, array, indicies, values, expected):
        '''
        Adds values to an array at specified indices.
        '''
        # 设置后端
        bm.set_backend(backend)
        # 转换为对应的后端数组
        array = bm.from_numpy(array)
        indicies = bm.from_numpy(indicies)
        values = bm.from_numpy(values)
        expected = bm.from_numpy(expected)
        # 测试
        assert bm.all(bm.add_at(array, indicies, values) == expected)
        
if __name__ == '__main__':
    pytest.main(['test/backend/test_non_standard.py', '-qs', '--disable-warnings'])