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
    from lib.data import (
        load_test_data,
        get_feature_columns,
        validate_data,
        ensure_lagged_feature_parity,
    )
except ImportError:
    # 如果lib.data导入失败，尝试直接导入
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
    from data import load_test_data, get_feature_columns, validate_data, ensure_lagged_feature_parity


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
    
    enriched = ensure_lagged_feature_parity(test_data)
    feature_cols = get_feature_columns(enriched)
    
    # 验证返回的特征列
    assert 'M1' in feature_cols
    assert 'M2' in feature_cols
    assert 'forward_returns' not in feature_cols  # 目标变量应该被排除
    assert 'date_id' not in feature_cols  # ID列应该被排除
    assert 'lagged_forward_returns' in feature_cols
    assert 'lagged_market_forward_excess_returns' in feature_cols
    

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


def test_ensure_lagged_feature_parity_aligns_with_date_order():
    df = pd.DataFrame(
        {
            'date_id': [2, 1, 3],
            'forward_returns': [0.02, -0.01, 0.05],
            'risk_free_rate': [0.001, 0.002, 0.003],
            'market_forward_excess_returns': [0.01, 0.015, 0.005],
        }
    )

    enriched = ensure_lagged_feature_parity(df)

    # 滞后列应该存在
    assert 'lagged_forward_returns' in enriched.columns
    assert 'lagged_market_forward_excess_returns' in enriched.columns

    # date_id=1（原index=1）没有前一天，应该为NaN
    assert np.isnan(enriched.loc[1, 'lagged_forward_returns'])
    # date_id=2（原index=0）的滞后值应等于date_id=1的forward_returns
    assert enriched.loc[0, 'lagged_forward_returns'] == pytest.approx(df.loc[1, 'forward_returns'])
    # date_id=3（原index=2）的滞后值应等于date_id=2的forward_returns
    assert enriched.loc[2, 'lagged_forward_returns'] == pytest.approx(df.loc[0, 'forward_returns'])


if __name__ == "__main__":
    test_get_feature_columns()
    test_validate_data()
    print("✅ 所有数据测试通过")
