"""
简单的特征工程验证测试
"""

import numpy as np
import pandas as pd
import sys
import os

# 添加working目录到路径
working_dir = os.path.dirname(os.path.dirname(__file__))
if working_dir not in sys.path:
    sys.path.insert(0, working_dir)

try:
    from lib.features import FeaturePipeline
except ImportError as e:
    print(f"导入错误: {e}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"Python路径: {sys.path}")
    raise

# 创建简单测试数据
np.random.seed(42)
train_data = pd.DataFrame({
    'date_id': range(100),
    'M1': np.random.randn(100),
    'M2': np.random.randn(100),
    'P1': np.random.randn(100) + 10,
    'P2': np.random.randn(100) + 10,
    'V1': np.random.randn(100),
    'E1': np.random.randn(100),
    'MOM1': np.random.randn(100),
    'lagged_forward_returns': np.random.randn(100),
    'forward_returns': np.random.randn(100) * 0.01,
})

print("创建测试数据完成")
print(f"数据形状: {train_data.shape}")

# 测试基础管道
print("\n测试基础FeaturePipeline...")
try:
    basic_pipeline = FeaturePipeline(extra_group_stats=False)
    basic_features = basic_pipeline.fit_transform(train_data)
    print(f"✅ 基础管道成功: {basic_features.shape}")
except Exception as e:
    print(f"❌ 基础管道失败: {e}")
    raise

# 测试增强管道
print("\n测试增强FeaturePipeline...")
try:
    enhanced_pipeline = FeaturePipeline(
        extra_group_stats=True,
        enable_data_quality=True,
        enable_feature_stability=True,
        outlier_detection=True
    )
    enhanced_features = enhanced_pipeline.fit_transform(train_data)
    print(f"✅ 增强管道成功: {enhanced_features.shape}")
    
    # 检查数据质量报告
    if hasattr(enhanced_pipeline, 'data_quality_metrics'):
        quality_report = enhanced_pipeline.get_data_quality_report()
        print(f"✅ 数据质量报告生成: {len(quality_report['quality_metrics'])} 个特征")
    
    # 检查稳定特征
    if hasattr(enhanced_pipeline, 'feature_stability_scores'):
        stable_features = enhanced_pipeline.get_stable_features()
        print(f"✅ 稳定特征: {len(stable_features)} 个")
        
except Exception as e:
    print(f"❌ 增强管道失败: {e}")
    raise

print("\n🎉 特征工程验证完成！")
print(f"基础特征: {basic_features.shape[1]}")
print(f"增强特征: {enhanced_features.shape[1]}")
print(f"特征增加: {enhanced_features.shape[1] - basic_features.shape[1]}")