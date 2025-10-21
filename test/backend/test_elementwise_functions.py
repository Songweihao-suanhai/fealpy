import pytest
from fealpy.backend import backend_manager as bm
from data_data_type_functions import *

class TestElementwiseFunctionsInterfaces:
    
    @pytest.mark.parametrize('backend', ['numpy', 'pytorch'])
    def test_sin(self, backend, backend_map):
        '''
        Calculates an implementation-dependent approximation to the sine for each element x_i of the input array x.
        '''
        # 设置后端
        bm.set_backend(backend)
        # 测试在不同后端下，`bm.sin` 指向正确的底层实现函数。
        module_backend = backend_map[backend]
        assert bm.sin is module_backend.sin
        
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    def test_asin(self, backend, backend_map):
        '''
	    Calculates an implementation-dependent approximation of the principal value of the inverse sine for each element x_i of the input array x.
        '''
        # 设置后端
        bm.set_backend(backend)
        # 测试在不同后端下，`bm.asin` 指向正确的底层实现函数。
        backend_module = backend_map[backend]
        assert bm.asin is backend_module.asin
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    def test_cos(self, backend, backend_map):
        '''
        Calculates an implementation-dependent approximation to the cosine for each element x_i of the input array x.
        '''
        # 设置后端
        bm.set_backend(backend)
        # 测试在不同后端下，`bm.cos` 指向正确的底层实现函数。
        backend_module = backend_map[backend]
        assert bm.cos is backend_module.cos
        
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    def test_acos(self, backend, backend_map):
        '''
        Calculates an implementation-dependent approximation of the principal value of the inverse cosine for each element x_i of the input array x.
        '''
        # 设置后端
        bm.set_backend(backend)
        # 测试在不同后端下，`bm.acos` 指向正确的底层实现函数。
        backend_module = backend_map[backend]
        assert bm.acos is backend_module.acos
        
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    def test_tan(self, backend, backend_map):
        '''
	    Calculates an implementation-dependent approximation to the tangent for each element x_i of the input array x.
        '''
        # 设置后端
        bm.set_backend(backend)
        # 测试在不同后端下，`bm.tan` 指向正确的底层实现函数。
        backend_module = backend_map[backend]
        assert bm.tan is backend_module.tan
        
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    def test_atan(self, backend, backend_map):
        '''
	    Calculates an implementation-dependent approximation of the principal value of the inverse tangent for each element x_i of the input array x.
        '''
        # 设置后端
        bm.set_backend(backend)
        # 测试在不同后端下，`bm.atan` 指向正确的底层实现函数。
        backend_module = backend_map[backend]
        assert bm.atan is backend_module.atan
        
if __name__ == '__main__':
    pytest.main(['test/backend/test_elementwise_functions.py', '-qs', '--disable-warnings'])