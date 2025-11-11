#!/usr/bin/env python3
"""
智能特征工程使用示例
展示如何使用新的智能特征选择、组合和优化功能
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
from working.lib.features import build_feature_pipeline, FeaturePipeline
from working.lib.data import load_train_data

def example_basic_usage():
    """基础使用示例"""
    print("📚 基础使用示例")
    print("="*50)
    
    # 加载数据
    df = load_train_data()
    print(f"📊 原始数据形状: {df.shape}")
    
    # 创建智能特征管道 - 推荐配置
    pipeline = build_feature_pipeline(
        enable_intelligent_selection=True,    # 启用智能特征选择
        feature_selection_method="mixed",     # 使用混合方法
        enable_feature_combinations=True,     # 启用特征组合
        combination_complexity=3,             # 中等复杂度
        enable_tiered_features=True,          # 启用分层特征
        enable_robust_scaling=True,           # 启用RobustScaler
        max_features=100,                     # 限制最终特征数
        standardize=True                      # 启用标准化
    )
    
    # 设置目标变量（用于特征选择）
    if 'market_forward_excess_returns' in df.columns:
        pipeline.target_column = 'market_forward_excess_returns'
    
    # 使用前500行进行演示
    demo_df = df.head(500).copy()
    
    # 拟合和转换
    features = pipeline.fit_transform(demo_df)
    
    print(f"✅ 处理完成！")
    print(f"   - 输出特征数: {features.shape[1]}")
    print(f"   - 数据样本数: {features.shape[0]}")
    print(f"   - 特征选择: {len(pipeline.selected_features) if pipeline.selected_features else 'N/A'} 个特征")
    
    return features, pipeline

def example_advanced_configuration():
    """高级配置示例"""
    print(f"\n🔧 高级配置示例")
    print("="*50)
    
    # 加载数据
    df = load_train_data()
    
    # 自定义配置：只使用相关性选择 + 简单特征组合
    pipeline_1 = build_feature_pipeline(
        enable_intelligent_selection=True,
        feature_selection_method="correlation",  # 只使用相关性
        enable_feature_combinations=True,
        combination_complexity=1,               # 基础组合
        enable_tiered_features=False,           # 关闭分层特征
        enable_robust_scaling=True,
        max_features=50
    )
    
    # 自定义配置：高复杂度组合 + RFE选择
    pipeline_2 = build_feature_pipeline(
        enable_intelligent_selection=True,
        feature_selection_method="rfe",         # 使用RFE
        enable_feature_combinations=True,
        combination_complexity=5,               # 最高复杂度
        enable_tiered_features=True,
        tiered_levels=6,                        # 细粒度分层
        enable_robust_scaling=True,
        max_features=80
    )
    
    # 测试不同配置
    demo_df = df.head(300).copy()
    if 'market_forward_excess_returns' in demo_df.columns:
        pipeline_1.target_column = 'market_forward_excess_returns'
        pipeline_2.target_column = 'market_forward_excess_returns'
    
    # 处理数据
    features_1 = pipeline_1.fit_transform(demo_df)
    features_2 = pipeline_2.fit_transform(demo_df)
    
    print(f"🔍 配置对比结果:")
    print(f"   配置1 (相关性+基础组合): {features_1.shape[1]} 特征")
    print(f"   配置2 (RFE+高级组合): {features_2.shape[1]} 特征")
    
    return features_1, features_2

def example_custom_pipeline():
    """自定义管道示例"""
    print(f"\n🎛️ 自定义管道示例")
    print("="*50)
    
    # 加载数据
    df = load_train_data()
    
    # 创建自定义管道
    pipeline = FeaturePipeline(
        enable_intelligent_selection=True,
        feature_selection_method="mixed",
        enable_feature_combinations=True,
        combination_complexity=4,
        enable_tiered_features=True,
        tiered_levels=4,
        enable_robust_scaling=True,
        max_features=60,
        standardize=True,
        clip_quantile=0.02,              # 裁剪极值
        outlier_detection=True,          # 异常值检测
        enable_data_quality=True,        # 数据质量分析
        enable_feature_stability=True    # 特征稳定性分析
    )
    
    # 设置目标
    demo_df = df.head(400).copy()
    if 'market_forward_excess_returns' in demo_df.columns:
        pipeline.target_column = 'market_forward_excess_returns'
    
    # 拟合和转换
    features = pipeline.fit_transform(demo_df)
    
    # 分析结果
    print(f"🔍 自定义管道分析:")
    print(f"   - 总特征数: {features.shape[1]}")
    print(f"   - 数据质量指标: {len(pipeline.data_quality_metrics)} 项")
    print(f"   - 稳定性分析: {'完成' if pipeline.feature_stability_scores else '未完成'}")
    print(f"   - 异常值边界: {len(pipeline.outlier_bounds)} 个特征")
    
    # 展示质量指标样例
    if pipeline.data_quality_metrics:
        sample_features = list(pipeline.data_quality_metrics.keys())[:3]
        print(f"\n📊 数据质量样例:")
        for feature in sample_features:
            metrics = pipeline.data_quality_metrics[feature]
            print(f"   {feature}: 缺失率={metrics.get('missing_rate', 0):.3f}, 偏度={metrics.get('skewness', 0):.3f}")
    
    return features, pipeline

def example_performance_monitoring():
    """性能监控示例"""
    print(f"\n📈 性能监控示例")
    print("="*50)
    
    # 加载数据
    df = load_train_data()
    
    # 启用监控的管道
    pipeline = build_feature_pipeline(
        enable_intelligent_selection=True,
        feature_selection_method="mixed",
        enable_feature_combinations=True,
        combination_complexity=3,
        enable_tiered_features=True,
        enable_robust_scaling=True,
        max_features=70,
        enable_data_quality=True,
        enable_feature_stability=True,
        outlier_detection=True
    )
    
    # 设置目标
    demo_df = df.head(350).copy()
    if 'market_forward_excess_returns' in demo_df.columns:
        pipeline.target_column = 'market_forward_excess_returns'
    
    import time
    start_time = time.time()
    features = pipeline.fit_transform(demo_df)
    processing_time = time.time() - start_time
    
    print(f"⏱️ 性能指标:")
    print(f"   - 处理时间: {processing_time:.3f} 秒")
    print(f"   - 特征数量: {features.shape[1]}")
    print(f"   - 样本数量: {features.shape[0]}")
    print(f"   - 特征/秒: {features.shape[1]/processing_time:.1f}")
    
    # 特征选择分析
    if pipeline.selected_features_meta:
        meta = pipeline.selected_features_meta
        selection_rate = meta['selected_count'] / meta['total_available'] * 100
        print(f"\n🎯 特征选择分析:")
        print(f"   - 选择率: {selection_rate:.1f}%")
        print(f"   - 选择方法: {meta['selection_method']}")
        print(f"   - 总可选特征: {meta['total_available']}")
        print(f"   - 实际选择: {meta['selected_count']}")
    
    # 数据质量分析
    if pipeline.feature_stability_scores:
        stability_scores = list(pipeline.feature_stability_scores.values())
        avg_stability = np.mean(stability_scores)
        stable_count = sum(1 for score in stability_scores if score > 0.5)
        
        print(f"\n📊 稳定性分析:")
        print(f"   - 平均稳定性: {avg_stability:.3f}")
        print(f"   - 稳定特征数: {stable_count}/{len(stability_scores)}")
        print(f"   - 稳定率: {stable_count/len(stability_scores)*100:.1f}%")
    
    return features, pipeline

def example_save_and_load():
    """保存和加载示例"""
    print(f"\n💾 保存和加载示例")
    print("="*50)
    
    # 加载数据
    df = load_train_data()
    
    # 创建和训练管道
    pipeline = build_feature_pipeline(
        enable_intelligent_selection=True,
        feature_selection_method="mixed",
        enable_feature_combinations=True,
        combination_complexity=2,
        enable_robust_scaling=True,
        max_features=50
    )
    
    demo_df = df.head(250).copy()
    if 'market_forward_excess_returns' in demo_df.columns:
        pipeline.target_column = 'market_forward_excess_returns'
    
    # 训练
    features = pipeline.fit_transform(demo_df)
    
    # 保存配置
    config = pipeline.to_config()
    print(f"💾 配置已保存，包含 {len(config)} 个参数")
    
    # 从配置重建管道
    new_pipeline = FeaturePipeline.from_config(config)
    print(f"🔄 管道配置重建完成")
    
    # 验证重建效果
    new_features = new_pipeline.transform(demo_df)
    
    print(f"✅ 验证结果:")
    print(f"   - 原始特征数: {features.shape[1]}")
    print(f"   - 重建特征数: {new_features.shape[1]}")
    print(f"   - 特征数匹配: {'✓' if features.shape[1] == new_features.shape[1] else '✗'}")
    
    return features, new_features

def main():
    """主函数"""
    print("🚀 智能特征工程使用示例")
    print("="*60)
    
    try:
        # 基础使用
        example_basic_usage()
        
        # 高级配置
        example_advanced_configuration()
        
        # 自定义管道
        example_custom_pipeline()
        
        # 性能监控
        example_performance_monitoring()
        
        # 保存和加载
        example_save_and_load()
        
        print(f"\n🎉 所有示例执行完成！")
        print(f"✨ 智能特征工程功能正常工作")
        
    except Exception as e:
        print(f"❌ 示例执行出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()