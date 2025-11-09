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
    from lib.features import (
        engineer_features,
        handle_missing_values,
        add_statistical_features,
        get_feature_groups,
        FeaturePipeline,
    )
    from lib.data import ensure_lagged_feature_parity
except ImportError:
    # 如果lib.features导入失败，尝试直接导入
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
    from features import engineer_features, handle_missing_values, add_statistical_features, get_feature_groups, FeaturePipeline
    from data import ensure_lagged_feature_parity


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

    assert result.shape[0] == 5
    assert result.shape[1] >= len(feature_cols)
    assert not result.isnull().any().any()


def test_engineer_features_can_return_pipeline_for_reuse():
    df = pd.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [5.0, 6.0, 7.0],
            "forward_returns": [0.0, 0.0, 0.0],
        }
    )

    (train_features, pipeline) = engineer_features(df, ["feature1", "feature2"], return_pipeline=True)
    assert pipeline is not None

    new_df = pd.DataFrame(
        {
            "feature1": [4.0, 5.0],
            "feature2": [8.0, 9.0],
        }
    )
    transformed = engineer_features(new_df, ["feature1", "feature2"], pipeline=pipeline)
    assert transformed.shape[1] == train_features.shape[1]
    

def test_get_feature_groups():
    """测试特征分组"""
    
    groups = get_feature_groups()
    
    # 验证分组结构
    expected_groups = ['market', 'economic', 'interest', 'price', 'volatility', 'sentiment', 'momentum', 'dummy']
    for group in expected_groups:
        assert group in groups
        assert isinstance(groups[group], list)
        assert len(groups[group]) > 0


def test_feature_pipeline_handles_small_samples_without_quantile_clipping():
    """验证在clip_quantile=0的小样本场景不会崩溃且产出无缺失值"""

    df = pd.DataFrame(
        {
            "feature_a": [1.0, np.nan, 3.0],
            "feature_b": [0.5, 0.6, np.nan],
            "forward_returns": [0.01, 0.02, 0.03],
        }
    )
    pipeline = FeaturePipeline(clip_quantile=0)
    transformed = pipeline.fit_transform(df[["feature_a", "feature_b"]])

    assert transformed.shape[0] == len(df)
    assert not transformed.isnull().any().any()


def test_feature_pipeline_preserves_schema_with_lagged_features():
    base_train = pd.DataFrame(
        {
            'date_id': [1, 2, 3, 4],
            'feature_a': [0.1, 0.2, 0.3, 0.4],
            'feature_b': [10, 11, 12, 13],
            'forward_returns': [0.01, -0.02, 0.03, -0.01],
            'risk_free_rate': [0.001, 0.001, 0.002, 0.002],
            'market_forward_excess_returns': [0.02, 0.01, 0.03, 0.00],
        }
    )
    train_df = ensure_lagged_feature_parity(base_train)

    pipeline = FeaturePipeline()
    train_features = pipeline.fit_transform(train_df)

    test_df = pd.DataFrame(
        {
            'date_id': [5, 6],
            'feature_a': [0.5, 0.6],
            'feature_b': [14, 15],
            'lagged_forward_returns': [0.03, -0.01],
            'lagged_market_forward_excess_returns': [0.02, 0.01],
            'lagged_risk_free_rate': [0.001, 0.001],
            'is_scored': [True, True],
        }
    )

    transformed_test = pipeline.transform(test_df)
    assert list(train_features.columns) == list(transformed_test.columns)


if __name__ == "__main__":
    test_handle_missing_values()
    test_add_statistical_features()
    test_engineer_features()
    test_get_feature_groups()
    print("✅ 所有特征测试通过")
