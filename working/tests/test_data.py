"""
测试数据加载和验证功能
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from lib.data import load_test_data, get_feature_columns, validate_data
except ImportError:
    # 如果lib.data导入失败，尝试直接导入
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
    from data import load_test_data, get_feature_columns, validate_data


def test_get_feature_columns():
    """测试特征列获取功能"""
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'date_id': [1, 2, 3],
        'M1': [0.1, 0.2, 0.3],
        'M2': [0.4, 0.5, 0.6],
        'forward_returns': [0.01, -0.02, 0.03],
        'risk_free_rate': [0.001, 0.001, 0.001],
        'market_forward_excess_returns': [0.009, -0.021, 0.029],
        'is_scored': [True, True, False]
    })
    
    feature_cols = get_feature_columns(test_data)
    
    # 验证返回的特征列
    assert 'M1' in feature_cols
    assert 'M2' in feature_cols
    assert 'forward_returns' not in feature_cols  # 目标变量应该被排除
    assert 'date_id' not in feature_cols  # ID列应该被排除
    

def test_validate_data():
    """测试数据验证功能"""
    
    # 测试有效数据
    valid_data = pd.DataFrame({
        'date_id': [1, 2, 3],
        'forward_returns': [0.01, -0.02, 0.03],
        'risk_free_rate': [0.001, 0.001, 0.001],
        'market_forward_excess_returns': [0.009, -0.021, 0.029]
    })
    
    assert validate_data(valid_data, "train") == True
    
    # 测试缺失列的数据
    invalid_data = pd.DataFrame({
        'date_id': [1, 2, 3]
    })
    
    # 对于训练数据应该失败
    assert validate_data(invalid_data, "train") == False
    
    # 对于测试数据应该通过（只需要date_id和is_scored）
    test_data_with_scored = pd.DataFrame({
        'date_id': [1, 2, 3],
        'is_scored': [True, True, False]
    })
    assert validate_data(test_data_with_scored, "test") == True


def test_data_loading_integration():
    """测试数据加载的集成测试"""
    
    # 这个测试需要实际的数据文件
    # 在测试环境中，我们跳过实际加载
    pass


if __name__ == "__main__":
    test_get_feature_columns()
    test_validate_data()
    print("✅ 所有数据测试通过")