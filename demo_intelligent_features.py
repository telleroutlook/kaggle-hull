#!/usr/bin/env python3
"""
智能特征工程功能展示
展示新的智能特征选择、组合和优化功能
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# 添加路径
sys.path.append('/home/dev/github/kaggle-hull')
sys.path.append('/home/dev/github/kaggle-hull/working')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from working.lib.features import FeaturePipeline, build_feature_pipeline
from working.lib.data import load_train_data, get_feature_columns

def demonstrate_feature_selection():
    """展示智能特征选择功能"""
    print("🎯 智能特征选择功能展示")
    print("="*50)
    
    # 加载数据
    df = load_train_data()
    
    if df is None:
        print("❌ 无法加载数据")
        return
    
    print(f"📊 原始数据信息:")
    print(f"  - 数据形状: {df.shape}")
    print(f"  - 特征列数: {len([col for col in df.columns if not col.startswith('forward_returns')])}")
    
    # 创建目标变量用于演示
    if 'market_forward_excess_returns' in df.columns:
        target_col = 'market_forward_excess_returns'
    else:
        df['demo_target'] = np.random.normal(0, 0.02, len(df))
        target_col = 'demo_target'
    
    # 智能特征选择演示
    pipeline = build_feature_pipeline(
        enable_intelligent_selection=True,
        feature_selection_method='mixed',
        max_features=40,
        enable_robust_scaling=True
    )
    pipeline.target_column = target_col
    
    # 使用前300行进行演示
    demo_df = df.head(300).copy()
    features = pipeline.fit_transform(demo_df)
    
    print(f"\n🔍 智能特征选择结果:")
    print(f"  - 输出特征数: {features.shape[1]}")
    print(f"  - 数据样本数: {features.shape[0]}")
    
    if pipeline.selected_features:
        print(f"  - 选定特征数量: {len(pipeline.selected_features)}")
        print(f"  - 特征选择方法: {pipeline.feature_selection_method}")
        print(f"  - 前10个选定特征: {pipeline.selected_features[:10]}")
    
    if pipeline.selected_features_meta:
        meta = pipeline.selected_features_meta
        print(f"\n📈 特征选择统计:")
        print(f"  - 可选特征总数: {meta['total_available']}")
        print(f"  - 实际选择数: {meta['selected_count']}")
        print(f"  - 选择比例: {meta['selected_count']/meta['total_available']*100:.1f}%")
    
    return features, pipeline

def demonstrate_feature_combinations():
    """展示特征组合功能"""
    print(f"\n🔄 智能特征组合功能展示")
    print("="*50)
    
    # 加载数据
    df = load_train_data()

    
    if df is None:
        print("❌ 无法加载数据")
        return None, None
    
    # 不同复杂度级别的特征组合
    complexity_levels = [1, 3, 5]
    results = {}
    
    for complexity in complexity_levels:
        pipeline = build_feature_pipeline(
            enable_feature_combinations=True,
            combination_complexity=complexity,
            max_features=20,
            enable_robust_scaling=False  # 关闭标准化以便更好观察组合效果
        )
        
        demo_df = df.head(200).copy()
        features = pipeline.fit_transform(demo_df)
        
        # 统计组合特征
        combinations = [col for col in features.columns if any(pattern in col for pattern in ['_x_', '_div_', '_squared', '_log', '_ewm_'])]
        basic_combos = [col for col in combinations if any(pattern in col for pattern in ['_x_', '_div_'])]
        advanced_combos = [col for col in combinations if any(pattern in col for pattern in ['_squared', '_log', '_ewm_'])]
        
        results[complexity] = {
            'total_features': features.shape[1],
            'combinations': len(combinations),
            'basic_combinations': len(basic_combos),
            'advanced_combinations': len(advanced_combos),
            'combination_ratio': len(combinations) / features.shape[1] * 100
        }
        
        print(f"🔢 复杂度级别 {complexity}:")
        print(f"  - 总特征数: {features.shape[1]}")
        print(f"  - 组合特征数: {len(combinations)}")
        print(f"  - 基础组合: {len(basic_combos)}")
        print(f"  - 高级组合: {len(advanced_combos)}")
        print(f"  - 组合占比: {len(combinations) / features.shape[1] * 100:.1f}%")
    
    return results, pipeline

def demonstrate_tiered_features():
    """展示分层特征工程"""
    print(f"\n🏗️ 分层特征工程展示")
    print("="*50)
    
    # 加载数据
    df = load_train_data()
    
    if df is None:
        print("❌ 无法加载数据")
        return None, None
    
    # 创建分层特征管道
    pipeline = build_feature_pipeline(
        enable_tiered_features=True,
        tiered_levels=4,
        max_features=25,
        enable_robust_scaling=False
    )
    
    demo_df = df.head(250).copy()
    features = pipeline.fit_transform(demo_df)
    
    # 统计分层特征
    tiered_features = [col for col in features.columns if any(pattern in col for pattern in ['_mean_', '_std_', 'low_vol', 'high_vol', 'strong_trend', 'weak_trend', 'bull_market', 'bear_market'])]
    
    # 按类型分类
    by_market_state = [col for col in tiered_features if any(pattern in col for pattern in ['_mean_', '_std_'])]
    by_volatility = [col for col in tiered_features if 'vol' in col]
    by_trend = [col for col in tiered_features if 'trend' in col]
    by_market_regime = [col for col in tiered_features if any(pattern in col for pattern in ['bull_market', 'bear_market'])]
    
    print(f"🏗️ 分层特征统计:")
    print(f"  - 总特征数: {features.shape[1]}")
    print(f"  - 分层特征数: {len(tiered_features)}")
    print(f"  - 状态分层: {len(by_market_state)}")
    print(f"  - 波动率分层: {len(by_volatility)}")
    print(f"  - 趋势分层: {len(by_trend)}")
    print(f"  - 市场形态: {len(by_market_regime)}")
    
    if tiered_features:
        print(f"\n📋 分层特征示例:")
        for i, feature in enumerate(tiered_features[:8]):
            print(f"  {i+1}. {feature}")
    
    return features, pipeline

def demonstrate_robust_scaling():
    """展示RobustScaler改进"""
    print(f"\n🔧 RobustScaler改进展示")
    print("="*50)
    
    # 加载数据
    df = load_train_data()
    
    if df is None:
        print("❌ 无法加载数据")
        return None, None
    
    # 创建带robust scaling的管道
    pipeline = build_feature_pipeline(
        standardize=True,
        enable_robust_scaling=True,
        max_features=15
    )
    
    demo_df = df.head(100).copy()
    features = pipeline.fit_transform(demo_df)
    
    print(f"🔧 标准化配置:")
    print(f"  - 启用标准化: {pipeline.standardize}")
    print(f"  - 启用RobustScaler: {pipeline.enable_robust_scaling}")
    
    if pipeline.scaler:
        print(f"  - 实际缩放器类型: {type(pipeline.scaler).__name__}")
        
        # 分析缩放效果
        numeric_features = features.select_dtypes(include=[np.number]).columns[:5]
        for col in numeric_features:
            original_std = features[col].std()
            mean_val = features[col].mean()
            print(f"  - {col}: 均值={mean_val:.3f}, 标准差={original_std:.3f}")
    else:
        print("  - 使用标准StandardScaler")
    
    print(f"  - 输出形状: {features.shape}")
    
    return features, pipeline

def demonstrate_data_quality():
    """展示数据质量分析"""
    print(f"\n📊 数据质量分析展示")
    print("="*50)
    
    # 加载数据
    df = load_train_data()

    
    if df is None:
        print("❌ 无法加载数据")
        return
    
    # 创建启用数据质量分析的管道
    pipeline = build_feature_pipeline(
        enable_data_quality=True,
        enable_feature_stability=True,
        outlier_detection=True,
        max_features=20
    )
    
    demo_df = df.head(150).copy()
    features = pipeline.fit_transform(demo_df)
    
    print(f"📊 数据质量报告:")
    print(f"  - 分析特征数: {len(pipeline.numeric_columns)}")
    print(f"  - 数据质量指标: {len(pipeline.data_quality_metrics)} 项")
    print(f"  - 稳定性分析: {'完成' if pipeline.feature_stability_scores else '未完成'}")
    print(f"  - 异常值检测: {len(pipeline.outlier_bounds)} 个特征已设置边界")
    
    # 展示质量指标样例
    if pipeline.data_quality_metrics:
        sample_features = list(pipeline.data_quality_metrics.keys())[:5]
        print(f"\n📈 质量指标示例:")
        for feature in sample_features:
            metrics = pipeline.data_quality_metrics[feature]
            print(f"  {feature}:")
            print(f"    - 缺失率: {metrics.get('missing_rate', 0):.3f}")
            print(f"    - 零值率: {metrics.get('zero_rate', 0):.3f}")
            print(f"    - 偏度: {metrics.get('skewness', 0):.3f}")
    
    # 展示稳定性得分
    if pipeline.feature_stability_scores:
        stable_features = [col for col, score in pipeline.feature_stability_scores.items() if score > 0.5]
        risky_features = [col for col, score in pipeline.feature_stability_scores.items() if score < 0.3]
        print(f"\n📈 稳定性分析:")
        print(f"  - 稳定特征: {len(stable_features)} 个")
        print(f"  - 不稳定特征: {len(risky_features)} 个")
        
        if stable_features:
            print(f"  - 稳定特征示例: {stable_features[:3]}")
        if risky_features:
            print(f"  - 不稳定特征示例: {risky_features[:3]}")

def create_performance_report():
    """创建性能对比报告"""
    print(f"\n📈 智能特征工程性能报告")
    print("="*60)
    
    # 加载数据
    df = load_train_data()
    
    if df is None:
        print("❌ 无法加载数据")
        return
    
    demo_df = df.head(200).copy()
    
    # 定义测试方案
    test_scenarios = {
        "基础版本": {
            "config": {
                "enable_intelligent_selection": False,
                "enable_feature_combinations": False,
                "enable_tiered_features": False,
                "enable_robust_scaling": False,
                "max_features": 25
            }
        },
        "智能选择": {
            "config": {
                "enable_intelligent_selection": True,
                "feature_selection_method": "mixed",
                "enable_feature_combinations": False,
                "enable_tiered_features": False,
                "enable_robust_scaling": True,
                "max_features": 25
            }
        },
        "特征组合": {
            "config": {
                "enable_intelligent_selection": True,
                "enable_feature_combinations": True,
                "combination_complexity": 3,
                "enable_tiered_features": False,
                "enable_robust_scaling": True,
                "max_features": 25
            }
        },
        "完整智能": {
            "config": {
                "enable_intelligent_selection": True,
                "feature_selection_method": "mixed",
                "enable_feature_combinations": True,
                "combination_complexity": 3,
                "enable_tiered_features": True,
                "tiered_levels": 4,
                "enable_robust_scaling": True,
                "max_features": 25
            }
        }
    }
    
    results = {}
    
    for scenario_name, scenario_config in test_scenarios.items():
        try:
            import time
            
            pipeline = build_feature_pipeline(**scenario_config["config"])
            start_time = time.time()
            features = pipeline.fit_transform(demo_df)
            processing_time = time.time() - start_time
            
            # 计算特征质量指标
            unique_features = features.shape[1]
            numeric_features = len(features.select_dtypes(include=[np.number]).columns)
            
            # 检查特征稳定性
            stability_score = 0
            if pipeline.feature_stability_scores:
                stability_score = np.mean(list(pipeline.feature_stability_scores.values()))
            
            results[scenario_name] = {
                'processing_time': processing_time,
                'feature_count': unique_features,
                'numeric_features': numeric_features,
                'stability_score': stability_score,
                'scalable': pipeline.scaler is not None if pipeline.standardize else True
            }
            
        except Exception as e:
            results[scenario_name] = {
                'error': str(e),
                'feature_count': 0,
                'processing_time': float('inf')
            }
    
    # 展示对比结果
    print(f"{'方案':<12} {'特征数':<8} {'时间(s)':<10} {'稳定性':<8} {'缩放器':<8}")
    print("-" * 50)
    
    for scenario, result in results.items():
        if 'error' not in result:
            features = result['feature_count']
            time_taken = result['processing_time']
            stability = result['stability_score']
            scaler = "Yes" if result['scalable'] else "No"
            
            print(f"{scenario:<12} {features:<8} {time_taken:<10.3f} {stability:<8.3f} {scaler:<8}")
        else:
            print(f"{scenario:<12} {'ERROR':<8} {'N/A':<10} {'N/A':<8} {'N/A':<8}")
    
    # 性能分析
    if "完整智能" in results and "基础版本" in results:
        intelligent = results["完整智能"]
        basic = results["基础版本"]
        
        if 'error' not in intelligent and 'error' not in basic:
            feature_improvement = (intelligent['feature_count'] - basic['feature_count']) / basic['feature_count'] * 100
            time_ratio = intelligent['processing_time'] / basic['processing_time']
            
            print(f"\n📊 性能提升分析:")
            print(f"  - 特征数量提升: {feature_improvement:+.1f}%")
            print(f"  - 处理时间比例: {time_ratio:.2f}x")
            print(f"  - 稳定性得分: {intelligent['stability_score']:.3f}")

def main():
    """主函数"""
    print("🚀 智能特征工程功能展示")
    print("="*60)
    
    # 展示各项功能
    try:
        demonstrate_feature_selection()
        demonstrate_feature_combinations()
        demonstrate_tiered_features()
        demonstrate_robust_scaling()
        demonstrate_data_quality()
        create_performance_report()
        
        print(f"\n🎉 智能特征工程功能展示完成！")
        print(f"✨ 主要改进包括:")
        print(f"  • 智能特征选择算法（相关性、互信息、RFE、聚类）")
        print(f"  • 多种特征组合策略（基础、多项式、条件、时间序列、非线性）")
        print(f"  • 分层市场特征工程（波动率、趋势、市场状态）")
        print(f"  • RobustScaler和分位数标准化")
        print(f"  • 完整的数据质量分析")
        print(f"  • 自适应特征选择和优化")
        
    except Exception as e:
        print(f"❌ 展示过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
