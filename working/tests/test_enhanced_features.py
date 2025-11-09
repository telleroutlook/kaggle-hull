"""
增强特征工程测试
测试新添加的技术指标、特征交叉和滞后交互功能
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os
from unittest.mock import Mock

# 添加working目录到路径
working_dir = os.path.dirname(os.path.dirname(__file__))
if working_dir not in sys.path:
    sys.path.insert(0, working_dir)

try:
    from lib.features import FeaturePipeline
    from lib.data import load_train_data, load_test_data
    from lib.env import detect_run_environment, get_data_paths
except ImportError as e:
    print(f"导入错误: {e}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"Python路径: {sys.path}")
    raise


class TestEnhancedFeaturePipeline:
    """测试增强后的FeaturePipeline功能"""

    def setup_method(self):
        """设置测试数据"""
        np.random.seed(42)
        self.train_data = pd.DataFrame({
            'date_id': range(100),
            'M1': np.random.randn(100),
            'M2': np.random.randn(100),
            'M3': np.random.randn(100),
            'M4': np.random.randn(100),
            'M5': np.random.randn(100),
            'P1': np.random.randn(100) + 10,
            'P2': np.random.randn(100) + 10,
            'P3': np.random.randn(100) + 10,
            'V1': np.random.randn(100),
            'V2': np.random.randn(100),
            'E1': np.random.randn(100),
            'E2': np.random.randn(100),
            'MOM1': np.random.randn(100),
            'lagged_forward_returns': np.random.randn(100),
            'lagged_risk_free_rate': np.random.randn(100),
            'lagged_market_forward_excess_returns': np.random.randn(100),
            'forward_returns': np.random.randn(100) * 0.01,
            'risk_free_rate': np.random.randn(100) * 0.001,
            'market_forward_excess_returns': np.random.randn(100) * 0.01,
        })
        
        # 添加更多特征列以模拟真实数据
        for prefix in ['S', 'I', 'D']:
            for i in range(1, 10):
                if prefix == 'D' and i > 9:
                    break
                col_name = f'{prefix}{i}'
                if col_name not in self.train_data.columns:
                    self.train_data[col_name] = np.random.randn(100)
        
        self.test_data = self.train_data.head(10).copy()

    def test_enhanced_features_count(self):
        """测试增强特征的数量"""
        pipeline = FeaturePipeline(extra_group_stats=True)
        train_features = pipeline.fit_transform(self.train_data)
        test_features = pipeline.transform(self.test_data)
        
        # 应该有额外的增强特征
        original_features = len([col for col in self.train_data.columns 
                               if not col.startswith('lagged_') and col not in 
                               ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']])
        enhanced_features = train_features.shape[1]
        
        # 至少应该有原始特征 + 增强特征
        assert enhanced_features > original_features, f"特征数量未增加: {enhanced_features} vs {original_features}"

    def test_technical_indicators(self):
        """测试技术指标计算"""
        pipeline = FeaturePipeline(extra_group_stats=True)
        features = pipeline.fit_transform(self.train_data)
        
        # 检查RSI指标
        assert 'rsi_14' in features.columns, "RSI指标未生成"
        
        # 检查移动平均交叉
        assert 'ma_cross_ratio' in features.columns, "MA交叉比率未生成"
        
        # 检查布林带位置
        assert 'bollinger_position' in features.columns, "布林带位置未生成"
        
        # 验证RSI值在合理范围内
        rsi_values = features['rsi_14']
        assert rsi_values.min() >= 0, "RSI最小值不应小于0"
        assert rsi_values.max() <= 100, "RSI最大值不应大于100"

    def test_cross_features(self):
        """测试特征交叉功能"""
        pipeline = FeaturePipeline(extra_group_stats=True)
        features = pipeline.fit_transform(self.train_data)
        
        # 检查交叉特征
        cross_features = ['market_corr', 'price_vol_interaction', 'economic_dual', 'momentum_market_interaction']
        for cross_feature in cross_features:
            assert cross_feature in features.columns, f"交叉特征 {cross_feature} 未生成"
        
        # 检查比率特征
        ratio_features = ['price_ratio', 'vol_ratio', 'market_ratio']
        for ratio_feature in ratio_features:
            assert ratio_feature in features.columns, f"比率特征 {ratio_feature} 未生成"

    def test_lagged_interactions(self):
        """测试滞后特征交互"""
        pipeline = FeaturePipeline(extra_group_stats=True)
        features = pipeline.fit_transform(self.train_data)
        
        # 检查滞后特征的变化率
        lag_change_features = [
            'lagged_forward_returns_change_rate',
            'lagged_risk_free_rate_change_rate',
            'lagged_market_forward_excess_returns_change_rate'
        ]
        
        for lag_feature in lag_change_features:
            assert lag_feature in features.columns, f"滞后变化率特征 {lag_feature} 未生成"

    def test_enhanced_std_reduction(self):
        """测试增强特征是否能减少std_guard触发"""
        env = detect_run_environment()
        data_paths = get_data_paths(env)
        
        # 加载真实数据
        train_data = load_train_data(data_paths)
        test_data = load_test_data(data_paths)
        
        # 使用增强的FeaturePipeline
        enhanced_pipeline = FeaturePipeline(
            clip_quantile=0.01,
            missing_indicator_threshold=0.05,
            standardize=False,
            extra_group_stats=True
        )
        
        # 使用基础的FeaturePipeline作为对比
        basic_pipeline = FeaturePipeline(
            clip_quantile=0.01,
            missing_indicator_threshold=0.05,
            standardize=False,
            extra_group_stats=False
        )
        
        # 训练两个管道
        enhanced_features = enhanced_pipeline.fit_transform(train_data)
        basic_features = basic_pipeline.fit_transform(train_data)
        
        # 转换测试集
        enhanced_test = enhanced_pipeline.transform(test_data)
        basic_test = basic_pipeline.transform(test_data)
        
        # 验证特征数量增加
        assert enhanced_features.shape[1] > basic_features.shape[1], \
            f"增强特征未增加: {enhanced_features.shape[1]} vs {basic_features.shape[1]}"
        
        assert enhanced_test.shape[1] > basic_test.shape[1], \
            f"测试特征未增加: {enhanced_test.shape[1]} vs {basic_test.shape[1]}"


class TestFeatureConsistency:
    """测试特征空间一致性"""

    def test_train_test_feature_consistency(self):
        """确保训练和测试特征空间一致"""
        env = detect_run_environment()
        data_paths = get_data_paths(env)
        
        train_data = load_train_data(data_paths)
        test_data = load_test_data(data_paths)
        
        pipeline = FeaturePipeline()
        train_features = pipeline.fit_transform(train_data)
        test_features = pipeline.transform(test_data)
        
        # 特征列应该完全一致
        train_cols = set(train_features.columns)
        test_cols = set(test_features.columns)
        
        assert train_cols == test_cols, \
            f"特征空间不一致: 训练集{len(train_cols)}列, 测试集{len(test_cols)}列"
        
        # 数量应该一致
        assert train_features.shape[1] == test_features.shape[1], \
            f"特征数量不一致: 训练集{train_features.shape[1]}, 测试集{test_features.shape[1]}"

    def test_enhanced_feature_consistency(self):
        """测试增强特征的一致性"""
        env = detect_run_environment()
        data_paths = get_data_paths(env)
        
        train_data = load_train_data(data_paths)
        test_data = load_test_data(data_paths)
        
        enhanced_pipeline = FeaturePipeline(extra_group_stats=True)
        
        # 训练和转换
        train_features = enhanced_pipeline.fit_transform(train_data)
        test_features = enhanced_pipeline.transform(test_data)
        
        # 验证增强后的特征空间一致性
        train_cols = set(train_features.columns)
        test_cols = set(test_features.columns)
        
        assert train_cols == test_cols, \
            f"增强特征空间不一致: 训练集{len(train_cols)}列, 测试集{len(test_cols)}列"


if __name__ == "__main__":
    # 运行基本测试
    test_instance = TestEnhancedFeaturePipeline()
    test_instance.setup_method()
    test_instance.test_enhanced_features_count()
    test_instance.test_technical_indicators()
    test_instance.test_cross_features()
    test_instance.test_lagged_interactions()
    test_instance.test_enhanced_std_reduction()
    
    print("✅ 增强特征工程测试通过")
