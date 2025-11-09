#!/usr/bin/env python3
"""Debug script to analyze feature preprocessing issues"""

import sys
import os
sys.path.insert(0, '/home/dev/github/kaggle-hull/working')

import pandas as pd
import numpy as np
from lib.data import load_train_data
from lib.features import FeaturePipeline

# 加载数据
print("=== 加载数据 ===")
train_df = load_train_data()
target = train_df["forward_returns"]

# 原始特征统计
print("\n=== 原始特征统计 ===")
feature_cols = [col for col in train_df.columns if not col in ['date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns']]
original_features = train_df[feature_cols]
print(f"原始特征数量: {len(feature_cols)}")
print(f"原始特征统计:")
print(f"  总体std范围: {original_features.std().min():.6f} - {original_features.std().max():.6f}")
print(f"  特征std中位数: {original_features.std().median():.6f}")

# 检查lagged特征
lagged_features = [col for col in feature_cols if 'lagged' in col]
print(f"\nLagged特征:")
for col in lagged_features:
    print(f"  {col}: min={train_df[col].min():.6f}, max={train_df[col].max():.6f}, std={train_df[col].std():.6f}")

# 特征工程前10个特征的分布
print(f"\n=== 原始特征样本 (前10个特征) ===")
for col in feature_cols[:10]:
    if col in train_df.columns:
        series = train_df[col]
        print(f"{col}: min={series.min():.6f}, max={series.max():.6f}, std={series.std():.6f}, null_rate={series.isnull().mean():.4f}")

# 特征工程后统计
print(f"\n=== 特征工程后统计 ===")
pipeline = FeaturePipeline()
features = pipeline.fit_transform(train_df[feature_cols])
print(f"工程后特征数量: {features.shape[1]}")
print(f"工程后特征统计:")
print(f"  总体std范围: {features.std().min():.6f} - {features.std().max():.6f}")
print(f"  特征std中位数: {features.std().median():.6f}")

# 显示前5个特征的分布
print(f"\n=== 工程后特征样本 (前5个特征) ===")
for i, col in enumerate(features.columns[:5]):
    series = features[col]
    print(f"{col}: min={series.min():.6f}, max={series.max():.6f}, std={series.std():.6f}")

# 相关性分析
print(f"\n=== 目标相关性分析 ===")
correlations = features.corrwith(target).abs().sort_values(ascending=False)
print(f"与目标相关性最高的前10个特征:")
for i, (feature, corr) in enumerate(correlations.head(10).items()):
    print(f"  {i+1}. {feature}: {corr:.4f}")

# lagged特征相关性
print(f"\nLagged特征相关性:")
lagged_corr = correlations[correlations.index.str.contains('lagged')]
for feature, corr in lagged_corr.head(10).items():
    print(f"  {feature}: {corr:.4f}")

# 检查特征是否被过度标准化
print(f"\n=== 标准化检查 ===")
print(f"特征值范围 [-3, 3] 的比例: {((features >= -3) & (features <= 3)).all(axis=1).mean():.4f}")
print(f"特征值范围 [-1, 1] 的比例: {((features >= -1) & (features <= 1)).all(axis=1).mean():.4f}")

# 检查最极端的预测值
print(f"\n=== 目标变量分析 ===")
print(f"目标变量统计:")
print(f"  非零值比例: {(target != 0).mean():.4f}")
print(f"  绝对值小于0.001的比例: {(abs(target) < 0.001).mean():.4f}")
print(f"  绝对值大于0.01的比例: {(abs(target) > 0.01).mean():.4f}")

# 查看特征工程参数
print(f"\n=== 特征工程参数 ===")
print(f"clip_quantile: {pipeline.clip_quantile}")
print(f"missing_indicator_threshold: {pipeline.missing_indicator_threshold}")
print(f"standardize: {pipeline.standardize}")
print(f"dtype: {pipeline.dtype}")
print(f"extra_group_stats: {pipeline.extra_group_stats}")