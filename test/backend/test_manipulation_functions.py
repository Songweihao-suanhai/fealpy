import pytest
from fealpy.backend import backend_manager as bm
from data_manipulation_functions import *

class TestManipulationFunctionsInterfaces:
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("array,repeat_counts,aixs,expected", repeat_test_data)
    def test_repeat(self, backend, array, repeat_counts, aixs, expected):
        '''
	    Repeats each element of an array a specified number of times on a per-element basis.
        '''
        # 设置后端
        bm.set_backend(backend)
        # 转换为对应的后端数组
        array = bm.from_numpy(array)
        expected = bm.from_numpy(expected)
        if isinstance(repeat_counts, np.ndarray):
            repeat_counts = bm.from_numpy(repeat_counts)
        # 测试
        assert bm.all((bm.repeat(array, repeat_counts, axis=aixs) == expected))
        
        
if __name__ == '__main__':
    pytest.main(['test/backend/test_manipulation_functions.py', '-qs', '--disable-warnings'])