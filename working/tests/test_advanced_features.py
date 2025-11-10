"""
高级特征工程测试
测试新添加的增强技术指标、分层统计、宏观交互和数据质量功能
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


class TestAdvancedFeaturePipeline:
    """测试增强后的FeaturePipeline高级功能"""

    def setup_method(self):
        """设置测试数据"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        
        # 创建更真实的测试数据
        self.train_data = pd.DataFrame({
            'date_id': range(200),
            'M1': np.random.randn(200) * 0.02,
            'M2': np.random.randn(200) * 0.02,
            'M3': np.random.randn(200) * 0.02,
            'M4': np.random.randn(200) * 0.02,
            'M5': np.random.randn(200) * 0.02,
            'P1': 100 + np.cumsum(np.random.randn(200) * 0.5),  # 价格序列
            'P2': 100 + np.cumsum(np.random.randn(200) * 0.5),
            'P3': 100 + np.cumsum(np.random.randn(200) * 0.5),
            'V1': np.random.exponential(0.02, 200),  # 波动率
            'V2': np.random.exponential(0.02, 200),
            'E1': np.random.randn(200) * 0.01,
            'E2': np.random.randn(200) * 0.01,
            'E3': np.random.randn(200) * 0.01,
            'S1': np.random.randn(200) * 0.5,
            'S2': np.random.randn(200) * 0.5,
            'I1': np.random.uniform(0.01, 0.05, 200),  # 利率
            'MOM1': np.random.randn(200) * 0.01,
            'MOM2': np.random.randn(200) * 0.01,
            'lagged_forward_returns': np.random.randn(200) * 0.01,
            'lagged_risk_free_rate': np.random.uniform(0.01, 0.05, 200),
            'lagged_market_forward_excess_returns': np.random.randn(200) * 0.01,
            'forward_returns': np.random.randn(200) * 0.01,
            'risk_free_rate': np.random.uniform(0.01, 0.05, 200),
            'market_forward_excess_returns': np.random.randn(200) * 0.01,
        })
        
        # 添加一些缺失值和异常值
        self.train_data.loc[10:15, 'M1'] = np.nan
        self.train_data.loc[30:35, 'P1'] = np.inf
        self.train_data.loc[50:55, 'V1'] = -1.0  # 负波动率
        
        self.test_data = self.train_data.head(20).copy()

    def test_advanced_technical_indicators(self):
        """测试高级技术指标"""
        pipeline = FeaturePipeline(
            extra_group_stats=True,
            enable_data_quality=True,
            enable_feature_stability=True
        )
        
        features = pipeline.fit_transform(self.train_data)
        
        # 检查Williams %R指标
        williams_features = [col for col in features.columns if 'williams_r_' in col]
        assert len(williams_features) > 0, "Williams %R指标未生成"
        
        # 检查Stochastic指标
        stoch_features = [col for col in features.columns if 'stoch_' in col]
        assert len(stoch_features) > 0, "Stochastic指标未生成"
        
        # 检查ADX指标
        adx_features = [col for col in features.columns if 'adx_' in col]
        assert len(adx_features) > 0, "ADX指标未生成"
        
        # 检查多期RSI
        rsi_features = [col for col in features.columns if 'rsi_' in col]
        assert len(rsi_features) >= 2, "多期RSI指标未生成"
        
        # 验证技术指标值在合理范围内
        if 'rsi_14' in features.columns:
            rsi_values = features['rsi_14']
            assert rsi_values.min() >= 0, f"RSI最小值超出范围: {rsi_values.min()}"
            assert rsi_values.max() <= 100, f"RSI最大值超出范围: {rsi_values.max()}"
        
        if 'williams_r_14' in features.columns:
            williams_values = features['williams_r_14']
            assert williams_values.min() >= -100, f"Williams %R最小值超出范围: {williams_values.min()}"
            assert williams_values.max() <= 0, f"Williams %R最大值超出范围: {williams_values.max()}"

    def test_tiered_statistics(self):
        """测试分层统计特征"""
        pipeline = FeaturePipeline(extra_group_stats=True)
        features = pipeline.fit_transform(self.train_data)
        
        # 检查分层统计特征
        tiered_features = [col for col in features.columns if any(suffix in col for suffix in ['_mean_low_vol', '_std_high_vol', '_mean_strong_trend'])]
        assert len(tiered_features) > 0, "分层统计特征未生成"
        
        # 验证分层特征值合理
        for col in tiered_features[:5]:  # 检查前5个分层特征
            values = features[col]
            assert values.notnull().sum() > 0, f"分层特征 {col} 全部为空"
            assert np.isfinite(values).all(), f"分层特征 {col} 包含非有限值"

    def test_macro_factor_interactions(self):
        """测试宏观因子交互特征"""
        pipeline = FeaturePipeline(extra_group_stats=True)
        features = pipeline.fit_transform(self.train_data)
        
        # 检查利率-市场交互
        rate_features = [col for col in features.columns if 'rate_adjusted' in col]
        assert len(rate_features) > 0, "利率-市场交互特征未生成"
        
        # 检查波动率-动量交互
        vol_momentum_features = [col for col in features.columns if 'vol_weighted_momentum' in col]
        assert len(vol_momentum_features) > 0, "波动率-动量交互特征未生成"
        
        # 检查宏观经济强度指标
        econ_features = [col for col in features.columns if 'economic_' in col]
        assert len(econ_features) > 0, "宏观经济指标未生成"

    def test_data_quality_analysis(self):
        """测试数据质量分析功能"""
        pipeline = FeaturePipeline(
            enable_data_quality=True,
            outlier_detection=True,
            missing_value_strategy="median"
        )
        
        features = pipeline.fit_transform(self.train_data)
        
        # 检查数据质量指标
        assert hasattr(pipeline, 'data_quality_metrics'), "数据质量指标未计算"
        assert hasattr(pipeline, 'feature_stability_scores'), "特征稳定性得分未计算"
        assert hasattr(pipeline, 'outlier_bounds'), "异常值边界未设置"
        
        # 验证数据质量报告
        quality_report = pipeline.get_data_quality_report()
        assert 'quality_metrics' in quality_report, "数据质量报告缺少质量指标"
        assert 'stability_scores' in quality_report, "数据质量报告缺少稳定性得分"
        
        # 检查稳定特征功能
        stable_features = pipeline.get_stable_features(threshold=0.1)
        assert isinstance(stable_features, list), "稳定特征应返回列表"
        
        risky_features = pipeline.get_risky_features(threshold=0.1)
        assert isinstance(risky_features, list), "风险特征应返回列表"

    def test_outlier_detection_and_handling(self):
        """测试异常值检测和处理"""
        pipeline = FeaturePipeline(
            outlier_detection=True,
            clip_quantile=0.01
        )
        
        features = pipeline.fit_transform(self.train_data)
        
        # 检查异常值边界
        assert len(pipeline.outlier_bounds) > 0, "异常值边界未设置"
        
        # 验证边界合理性
        for col, (lower, upper) in pipeline.outlier_bounds.items():
            assert lower < upper, f"异常值边界下界大于等于上界: {col}"
            assert np.isfinite(lower) and np.isfinite(upper), f"异常值边界包含非有限值: {col}"
        
        # 检查处理后的特征是否仍有异常值
        for col in features.select_dtypes(include=[np.number]).columns[:10]:
            if col in pipeline.outlier_bounds:
                lower, upper = pipeline.outlier_bounds[col]
                assert features[col].min() >= lower, f"特征 {col} 仍有低于边界的值"
                assert features[col].max() <= upper, f"特征 {col} 仍有高于边界的值"

    def test_smart_missing_value_handling(self):
        """测试智能缺失值处理"""
        # 创建包含大量缺失值的数据
        data_with_missing = self.train_data.copy()
        data_with_missing.loc[20:40, ['M1', 'M2', 'P1']] = np.nan
        data_with_missing.loc[50:60, 'V1'] = np.nan
        
        for strategy in ['median', 'mean', 'ffill', 'bfill']:
            pipeline = FeaturePipeline(
                missing_value_strategy=strategy,
                enable_data_quality=True
            )
            
            features = pipeline.fit_transform(data_with_missing)
            
            # 验证缺失值已处理
            for col in ['M1', 'M2', 'P1', 'V1']:
                if col in features.columns:
                    null_count = features[col].isnull().sum()
                    assert null_count == 0, f"使用策略 {strategy} 后特征 {col} 仍有缺失值: {null_count}"

    def test_enhanced_feature_space(self):
        """测试增强特征空间"""
        # 基础管道
        basic_pipeline = FeaturePipeline(
            extra_group_stats=False,
            enable_data_quality=False
        )
        
        # 增强管道
        enhanced_pipeline = FeaturePipeline(
            extra_group_stats=True,
            enable_data_quality=True,
            enable_feature_stability=True,
            outlier_detection=True
        )
        
        basic_features = basic_pipeline.fit_transform(self.train_data)
        enhanced_features = enhanced_pipeline.fit_transform(self.train_data)
        
        # 增强版本应该产生更多特征
        assert enhanced_features.shape[1] > basic_features.shape[1], \
            f"增强特征未增加: 基础{basic_features.shape[1]} vs 增强{enhanced_features.shape[1]}"
        
        # 检查特征命名的一致性
        enhanced_feature_names = set(enhanced_features.columns)
        
        # 检查一些预期的高级特征
        expected_patterns = [
            'rsi_', 'williams_r_', 'stoch_', 'adx_', 
            'macd_', 'bollinger_', 'tier_', 'economic_',
            'rate_adjusted', 'vol_weighted'
        ]
        
        found_patterns = 0
        for pattern in expected_patterns:
            matching_features = [col for col in enhanced_features.columns if pattern in col]
            if matching_features:
                found_patterns += 1
        
        assert found_patterns >= 5, f"高级特征模式匹配不足: {found_patterns}/{len(expected_patterns)}"

    def test_pipeline_config_serialization(self):
        """测试管道配置序列化"""
        pipeline = FeaturePipeline(
            extra_group_stats=True,
            enable_data_quality=True,
            enable_feature_stability=True,
            outlier_detection=True,
            missing_value_strategy="median"
        )
        
        # 获取配置
        config = pipeline.to_config()
        
        # 验证配置包含所有字段
        assert 'enable_data_quality' in config, "配置缺少数据质量开关"
        assert 'enable_feature_stability' in config, "配置缺少特征稳定性开关"
        assert 'outlier_detection' in config, "配置缺少异常值检测开关"
        assert 'missing_value_strategy' in config, "配置缺少缺失值处理策略"
        
        # 从配置重新创建管道
        new_pipeline = FeaturePipeline.from_config(config)
        
        # 验证配置一致性
        assert new_pipeline.enable_data_quality == pipeline.enable_data_quality
        assert new_pipeline.enable_feature_stability == pipeline.enable_feature_stability
        assert new_pipeline.outlier_detection == pipeline.outlier_detection
        assert new_pipeline.missing_value_strategy == pipeline.missing_value_strategy


if __name__ == "__main__":
    # 运行高级特征工程测试
    test_instance = TestAdvancedFeaturePipeline()
    test_instance.setup_method()
    
    print("运行高级特征工程测试...")
    test_instance.test_advanced_technical_indicators()
    print("✅ 高级技术指标测试通过")
    
    test_instance.test_tiered_statistics()
    print("✅ 分层统计特征测试通过")
    
    test_instance.test_macro_factor_interactions()
    print("✅ 宏观因子交互测试通过")
    
    test_instance.test_data_quality_analysis()
    print("✅ 数据质量分析测试通过")
    
    test_instance.test_outlier_detection_and_handling()
    print("✅ 异常值检测和处理测试通过")
    
    test_instance.test_smart_missing_value_handling()
    print("✅ 智能缺失值处理测试通过")
    
    test_instance.test_enhanced_feature_space()
    print("✅ 增强特征空间测试通过")
    
    test_instance.test_pipeline_config_serialization()
    print("✅ 管道配置序列化测试通过")
    
    print("\n🎉 所有高级特征工程测试通过！")
