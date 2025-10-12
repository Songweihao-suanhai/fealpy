import pytest
from fealpy.backend import backend_manager as bm
from data_data_type_functions import *

class TestDataTypeFunctionsInterfaces:
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("array,expected_dtype,expected", astype_test_data)
    def test_astype(self, backend, dtype_map, array, expected_dtype, expected):
        '''
        Copies an array to a specified data type irrespective of Type Promotion Rules rules.
        '''
        # # pytorch 不支持字符串类型，跳过
        # if backend == 'pytorch' and is_str:
        #     pytest.skip("PyTorch backend does not support string types")
        # 设置后端
        bm.set_backend(backend)
        # 转换为后端数组
        array = bm.from_numpy(array)
        expected = bm.from_numpy(expected)
        expected_dtype = dtype_map[backend][expected_dtype]
        # 测试
        assert bm.all((bm.astype(array, expected_dtype) == expected))
        
if __name__ == '__main__':
    pytest.main(['test/backend/test_data_type_functions.py', '-qs', '--disable-warnings'])