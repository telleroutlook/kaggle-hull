#!/usr/bin/env python3
"""
智能特征工程测试脚本
测试新的特征选择、组合和优化功能
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
from working.lib.features import FeaturePipeline, build_feature_pipeline
from working.lib.data import load_train_data, get_feature_columns

def test_intelligent_feature_selection():
    """测试智能特征选择功能"""
    print("🧪 测试智能特征选择功能...")
    
    # 加载数据
    df = load_train_data()
    
    if df is None:
        print("❌ 无法加载数据")
        return False
    
    print(f"📊 原始数据形状: {df.shape}")
    
    # 创建智能特征管道
    pipeline = build_feature_pipeline(
        enable_intelligent_selection=True,
        enable_feature_combinations=True,
        enable_tiered_features=True,
        enable_robust_scaling=True,
        feature_selection_method='mixed',
        combination_complexity=3,
        max_features=50  # 限制特征数量用于测试
    )
    
    # 模拟目标变量（用于特征选择）
    if 'market_forward_excess_returns' in df.columns:
        pipeline.target_column = 'market_forward_excess_returns'
    else:
        # 创建模拟目标
        df['target'] = np.random.normal(0, 0.02, len(df))
        pipeline.target_column = 'target'
    
    # 拟合和转换
    try:
        features = pipeline.fit_transform(df)
        print(f"✅ 特征转换成功，输出形状: {features.shape}")
        
        # 检查特征选择结果
        if pipeline.selected_features:
            print(f"🎯 选择了 {len(pipeline.selected_features)} 个特征")
            print(f"📋 选择的特征示例: {pipeline.selected_features[:10]}")
        
        # 检查特征元数据
        if pipeline.selected_features_meta:
            meta = pipeline.selected_features_meta
            print(f"🔍 特征选择元数据: 总数={meta['total_available']}, 选择={meta['selected_count']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 特征选择失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_features():
    """测试增强特征功能"""
    print("\n🧪 测试增强特征功能...")
    
    # 加载数据
    df = load_train_data()
    
    if df is None:
        print("❌ 无法加载数据")
        return False
    
    # 创建基础特征管道
    pipeline = build_feature_pipeline(
        enable_enhanced_features=True,
        enable_feature_combinations=True,
        combination_complexity=4,
        max_features=30
    )
    
    try:
        # 只使用前100行数据进行快速测试
        test_df = df.head(100).copy()
        features = pipeline.fit_transform(test_df)
        
        print(f"✅ 增强特征生成成功，输出形状: {features.shape}")
        
        # 检查技术指标
        technical_indicators = [col for col in features.columns if any(tech in col.lower() for tech in ['rsi', 'macd', 'adx', 'bollinger'])]
        print(f"📈 技术指标数量: {len(technical_indicators)}")
        
        # 检查特征组合
        combinations = [col for col in features.columns if '_x_' in col or '_div_' in col]
        print(f"🔄 特征组合数量: {len(combinations)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 增强特征测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_tiered_features():
    """测试分层特征工程"""
    print("\n🧪 测试分层特征工程...")
    
    # 加载数据
    df = load_train_data()
    
    if df is None:
        print("❌ 无法加载数据")
        return False
    
    # 创建分层特征管道
    pipeline = build_feature_pipeline(
        enable_tiered_features=True,
        tiered_levels=4,
        max_features=25
    )
    
    try:
        # 只使用前150行数据进行测试
        test_df = df.head(150).copy()
        features = pipeline.fit_transform(test_df)
        
        print(f"✅ 分层特征生成成功，输出形状: {features.shape}")
        
        # 检查分层特征
        tiered_features = [col for col in features.columns if any(tier in col for tier in ['_mean_', '_std_', 'tier_', 'tiered_'])]
        print(f"🏗️ 分层特征数量: {len(tiered_features)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 分层特征测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_robust_scaling():
    """测试RobustScaler改进"""
    print("\n🧪 测试RobustScaler改进...")
    
    # 加载数据
    df = load_train_data()
    
    if df is None:
        print("❌ 无法加载数据")
        return False
    
    # 创建启用robust scaling的管道
    pipeline = build_feature_pipeline(
        standardize=True,
        enable_robust_scaling=True,
        max_features=20
    )
    
    try:
        test_df = df.head(50).copy()
        features = pipeline.fit_transform(test_df)
        
        print(f"✅ RobustScaler测试成功，输出形状: {features.shape}")
        
        # 检查缩放器
        if pipeline.scaler:
            print(f"🔧 使用的缩放器类型: {type(pipeline.scaler).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ RobustScaler测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_comparison():
    """测试性能对比"""
    print("\n🧪 测试性能对比...")
    
    # 加载数据
    df = load_train_data()
    
    if df is None:
        print("❌ 无法加载数据")
        return False
    
    # 使用前200行数据进行对比测试
    test_df = df.head(200).copy()
    
    # 标准管道
    standard_pipeline = build_feature_pipeline(
        enable_feature_selection=False,
        enable_feature_combinations=False,
        enable_tiered_features=False,
        max_features=30
    )
    
    # 智能管道
    intelligent_pipeline = build_feature_pipeline(
        enable_intelligent_selection=True,
        enable_feature_combinations=True,
        enable_tiered_features=True,
        combination_complexity=3,
        max_features=30
    )
    
    try:
        import time
        
        # 测试标准管道
        start_time = time.time()
        standard_features = standard_pipeline.fit_transform(test_df)
        standard_time = time.time() - start_time
        
        # 测试智能管道
        start_time = time.time()
        intelligent_features = intelligent_pipeline.fit_transform(test_df)
        intelligent_time = time.time() - start_time
        
        print(f"📊 标准管道: {standard_features.shape} 特征, 耗时 {standard_time:.2f}秒")
        print(f"🧠 智能管道: {intelligent_features.shape} 特征, 耗时 {intelligent_time:.2f}秒")
        
        # 检查特征质量指标
        if intelligent_pipeline.feature_stability_scores:
            avg_stability = np.mean(list(intelligent_pipeline.feature_stability_scores.values()))
            print(f"📈 平均特征稳定性: {avg_stability:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 性能对比测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始智能特征工程测试...\n")
    
    tests = [
        ("智能特征选择", test_intelligent_feature_selection),
        ("增强特征功能", test_enhanced_features),
        ("分层特征工程", test_tiered_features),
        ("RobustScaler改进", test_robust_scaling),
        ("性能对比", test_performance_comparison)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试执行失败: {str(e)}")
            results.append((test_name, False))
    
    # 总结结果
    print("\n" + "="*60)
    print("📋 测试结果总结:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 总体结果: {passed}/{len(results)} 测试通过")
    
    if passed == len(results):
        print("🎉 所有智能特征工程功能测试通过！")
        return True
    else:
        print("⚠️ 部分功能需要进一步调试")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)