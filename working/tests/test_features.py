"""
测试特征工程功能
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from lib.features import engineer_features, handle_missing_values, add_statistical_features, get_feature_groups
except ImportError:
    # 如果lib.features导入失败，尝试直接导入
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
    from features import engineer_features, handle_missing_values, add_statistical_features, get_feature_groups


def test_handle_missing_values():
    """测试缺失值处理"""
    
    # 创建包含缺失值的测试数据
    df = pd.DataFrame({
        'numeric_col1': [1.0, np.nan, 3.0, 4.0],
        'numeric_col2': [5.0, 6.0, np.nan, 8.0],
        'categorical_col': ['A', 'B', np.nan, 'D']
    })
    
    result = handle_missing_values(df)
    
    # 验证没有缺失值
    assert not result.isnull().any().any()
    
    # 验证数值列用中位数填充
    assert result['numeric_col1'][1] == df['numeric_col1'].median()
    assert result['numeric_col2'][2] == df['numeric_col2'].median()
    
    # 验证分类列用众数填充
    assert result['categorical_col'][2] in ['A', 'B', 'D']


def test_add_statistical_features():
    """测试统计特征添加"""
    
    # 创建测试数据
    df = pd.DataFrame({
        'feature1': range(1, 21),  # 1到20
        'feature2': range(21, 41)  # 21到40
    })
    
    result = add_statistical_features(df)
    
    # 验证滚动特征添加
    assert 'feature1_rolling_mean_5' in result.columns
    assert 'feature1_rolling_std_5' in result.columns
    assert 'feature2_rolling_mean_10' in result.columns
    
    # 验证滞后特征添加
    assert 'feature1_lag_1' in result.columns
    assert 'feature2_lag_3' in result.columns
    
    # 验证滚动特征计算
    assert np.isclose(result['feature1_rolling_mean_5'][5], 3.0)  # 1+2+3+4+5 / 5 = 3
    
    # 验证滞后特征
    assert result['feature1_lag_1'][1] == 1  # 滞后1的值


def test_engineer_features():
    """测试特征工程完整流程"""
    
    # 创建测试数据
    df = pd.DataFrame({
        'feature1': [1.0, 2.0, np.nan, 4.0, 5.0],
        'feature2': [5.0, np.nan, 7.0, 8.0, 9.0],
        'date_id': [1, 2, 3, 4, 5],
        'forward_returns': [0.01, -0.02, 0.03, -0.01, 0.02]
    })
    
    feature_cols = ['feature1', 'feature2']
    result = engineer_features(df, feature_cols)
    
    # 验证结果
    assert result.shape[0] == 5  # 行数相同
    assert result.shape[1] > 2   # 特征数增加
    assert not result.isnull().any().any()  # 没有缺失值
    

def test_get_feature_groups():
    """测试特征分组"""
    
    groups = get_feature_groups()
    
    # 验证分组结构
    expected_groups = ['market', 'economic', 'interest', 'price', 'volatility', 'sentiment', 'momentum', 'dummy']
    for group in expected_groups:
        assert group in groups
        assert isinstance(groups[group], list)
        assert len(groups[group]) > 0


if __name__ == "__main__":
    test_handle_missing_values()
    test_add_statistical_features()
    test_engineer_features()
    test_get_feature_groups()
    print("✅ 所有特征测试通过")