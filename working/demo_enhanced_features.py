"""
特征工程增强演示脚本
展示新功能如何提升模型预测能力
"""

import numpy as np
import pandas as pd
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# 添加working目录到路径
working_dir = os.path.dirname(os.path.dirname(__file__))
if working_dir not in sys.path:
    sys.path.insert(0, working_dir)

from lib.features import FeaturePipeline
from lib.data import load_train_data, load_test_data
from lib.env import detect_run_environment, get_data_paths


def create_enhanced_demo_data():
    """创建演示用的增强数据集"""
    np.random.seed(42)
    n_samples = 200
    
    # 创建基础市场数据
    data = {
        'date_id': range(n_samples),
        'M1': np.random.randn(n_samples) * 0.02,
        'M2': np.random.randn(n_samples) * 0.02,
        'M3': np.random.randn(n_samples) * 0.02,
        'M4': np.random.randn(n_samples) * 0.02,
        'M5': np.random.randn(n_samples) * 0.02,
        'P1': 100 + np.cumsum(np.random.randn(n_samples) * 0.5),  # 价格序列
        'P2': 100 + np.cumsum(np.random.randn(n_samples) * 0.5),
        'P3': 100 + np.cumsum(np.random.randn(n_samples) * 0.5),
        'V1': np.random.exponential(0.02, n_samples),  # 波动率
        'V2': np.random.exponential(0.02, n_samples),
        'E1': np.random.randn(n_samples) * 0.01,
        'E2': np.random.randn(n_samples) * 0.01,
        'E3': np.random.randn(n_samples) * 0.01,
        'S1': np.random.randn(n_samples) * 0.5,
        'S2': np.random.randn(n_samples) * 0.5,
        'I1': np.random.uniform(0.01, 0.05, n_samples),  # 利率
        'MOM1': np.random.randn(n_samples) * 0.01,
        'MOM2': np.random.randn(n_samples) * 0.01,
        'lagged_forward_returns': np.random.randn(n_samples) * 0.01,
        'lagged_risk_free_rate': np.random.uniform(0.01, 0.05, n_samples),
        'lagged_market_forward_excess_returns': np.random.randn(n_samples) * 0.01,
        'forward_returns': np.random.randn(n_samples) * 0.01,
        'risk_free_rate': np.random.uniform(0.01, 0.05, n_samples),
        'market_forward_excess_returns': np.random.randn(n_samples) * 0.01,
    }
    
    return pd.DataFrame(data)


def demonstrate_enhanced_features():
    """演示增强特征工程功能"""
    
    print("🚀 特征工程增强演示")
    print("=" * 50)
    
    # 1. 创建演示数据
    print("\n📊 1. 创建演示数据")
    demo_data = create_enhanced_demo_data()
    print(f"   数据形状: {demo_data.shape}")
    print(f"   特征列: {len([col for col in demo_data.columns if not col.startswith(('date_id', 'forward', 'risk_free', 'market_forward', 'lagged_'))])}")
    
    # 2. 基础特征工程
    print("\n🔧 2. 基础特征工程")
    basic_pipeline = FeaturePipeline(
        extra_group_stats=False,
        enable_data_quality=False,
        enable_feature_stability=False,
        outlier_detection=False
    )
    
    basic_features = basic_pipeline.fit_transform(demo_data)
    print(f"   基础特征数量: {basic_features.shape[1]}")
    
    # 3. 增强特征工程
    print("\n⚡ 3. 增强特征工程")
    enhanced_pipeline = FeaturePipeline(
        extra_group_stats=True,
        enable_data_quality=True,
        enable_feature_stability=True,
        outlier_detection=True,
        missing_value_strategy="median",
        clip_quantile=0.01
    )
    
    enhanced_features = enhanced_pipeline.fit_transform(demo_data)
    print(f"   增强特征数量: {enhanced_features.shape[1]}")
    print(f"   特征增加: {enhanced_features.shape[1] - basic_features.shape[1]} (+{((enhanced_features.shape[1] - basic_features.shape[1]) / basic_features.shape[1] * 100):.1f}%)")
    
    # 4. 展示高级技术指标
    print("\n📈 4. 高级技术指标")
    tech_indicators = [
        'rsi_14', 'williams_r_14', 'stoch_k_14', 'stoch_d_14', 
        'adx_value', 'adx_plus_di', 'adx_minus_di', 'macd_12_26',
        'macd_signal_12_26', 'macd_hist_12_26', 'ma_cross_5_20', 
        'bollinger_width_20', 'vol_ratio_10'
    ]
    
    available_indicators = [ind for ind in tech_indicators if ind in enhanced_features.columns]
    print(f"   生成技术指标: {len(available_indicators)}")
    for indicator in available_indicators[:5]:
        values = enhanced_features[indicator]
        print(f"   - {indicator}: 均值={values.mean():.4f}, 标准差={values.std():.4f}")
    
    # 5. 展示分层统计特征
    print("\n📊 5. 分层统计特征")
    tiered_features = [col for col in enhanced_features.columns if any(suffix in col for suffix in ['_mean_low_vol', '_std_high_vol', '_mean_strong_trend'])]
    print(f"   生成分层统计特征: {len(tiered_features)}")
    
    # 6. 展示宏观因子交互
    print("\n🌍 6. 宏观因子交互")
    macro_features = [col for col in enhanced_features.columns if any(keyword in col for keyword in ['rate_adjusted', 'vol_weighted', 'economic_', 'sentiment_'])]
    print(f"   生成宏观交互特征: {len(macro_features)}")
    
    # 7. 数据质量分析
    print("\n🔍 7. 数据质量分析")
    quality_report = enhanced_pipeline.get_data_quality_report()
    quality_metrics = quality_report['quality_metrics']
    stability_scores = quality_report['stability_scores']
    
    print(f"   分析特征数量: {len(quality_metrics)}")
    print(f"   稳定性分析特征: {len(stability_scores)}")
    
    # 展示部分质量指标
    sample_features = list(quality_metrics.keys())[:3]
    for feature in sample_features:
        metrics = quality_metrics[feature]
        print(f"   - {feature}: 缺失率={metrics['missing_rate']:.3f}, 偏度={metrics['skewness']:.3f}")
    
    # 8. 特征选择
    print("\n🎯 8. 特征选择")
    stable_features = enhanced_pipeline.get_stable_features(threshold=0.1)
    risky_features = enhanced_pipeline.get_risky_features(threshold=0.1)
    
    print(f"   稳定特征 (阈值≥0.1): {len(stable_features)}")
    print(f"   风险特征 (阈值<0.1): {len(risky_features)}")
    
    # 9. 异常值处理
    print("\n🚨 9. 异常值处理")
    outlier_bounds = enhanced_pipeline.outlier_bounds
    print(f"   设置异常值边界: {len(outlier_bounds)} 个特征")
    
    # 10. 总结
    print("\n✅ 10. 总结")
    print(f"   基础特征 → 增强特征: {basic_features.shape[1]} → {enhanced_features.shape[1]}")
    print(f"   特征增长: +{enhanced_features.shape[1] - basic_features.shape[1]} 个特征")
    print(f"   技术指标: {len(available_indicators)} 个")
    print(f"   分层统计: {len(tiered_features)} 个")
    print(f"   宏观交互: {len(macro_features)} 个")
    print(f"   数据质量特征: ✅")
    print(f"   异常值处理: ✅")
    print(f"   特征稳定性分析: ✅")
    
    print("\n🎉 特征工程增强演示完成!")
    return enhanced_pipeline, enhanced_features


if __name__ == "__main__":
    try:
        pipeline, features = demonstrate_enhanced_features()
        print(f"\n📁 演示数据可从pipeline对象获取")
        print(f"   管道类型: {type(pipeline)}")
        print(f"   特征形状: {features.shape}")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()