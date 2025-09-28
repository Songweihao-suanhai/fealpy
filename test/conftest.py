import pytest
import numpy as np
import torch

BACKEND_MAP = {
    'numpy': np,
    'pytorch': torch
}

DTYPE_MAP = {
    'numpy': {
        # 整数类型
        'float32': np.float32,
        'float64': np.float64,
        'int8': np.int8,         # 8位整数
        'int16': np.int16,       # 16位整数
        'int32': np.int32,
        'int64': np.int64,
        'uint8': np.uint8,       # 无符号8位整数（常用于图像数据）
        'uint16': np.uint16,     # 无符号16位整数
        'uint32': np.uint32,     # 无符号32位整数
        'uint64': np.uint64,     # 无符号64位整数
        # 布尔和其他类型
        'bool': np.bool_,
        'complex64': np.complex64,  # 复数类型（32位）
        'complex128': np.complex128, # 复数类型（64位）
        'str': np.str_,          # 字符串类型
    },
    'pytorch': {
        # 整数类型
        'float32': torch.float32,
        'float64': torch.float64,
        'int8': torch.int8,
        'int16': torch.int16,
        'int32': torch.int32,
        'int64': torch.int64,
        'uint8': torch.uint8,    # 无符号8位整数（常用于图像数据）
        # PyTorch目前不支持uint16/32/64原生类型
        # 布尔和其他类型
        'bool': torch.bool,
        'complex64': torch.complex64,
        'complex128': torch.complex128,
        'str': None
    }
}

@pytest.fixture(scope="session")
def backend_map():
    return BACKEND_MAP

@pytest.fixture(scope="session")
def dtype_map():
    return DTYPE_MAP

