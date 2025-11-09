#!/usr/bin/env python3
"""Debug script to test without standardization"""

import sys
import os
sys.path.insert(0, '/home/dev/github/kaggle-hull/working')

import pandas as pd
import numpy as np
from lib.data import load_train_data
from lib.features import FeaturePipeline
from lib.models import HullModel

# 加载数据
print("=== 加载数据 ===")
train_df = load_train_data()

# 特征工程 - 不使用标准化
print("\n=== 特征工程 (无标准化) ===")
pipeline = FeaturePipeline(standardize=False)
features = pipeline.fit_transform(train_df)
print(f"特征矩阵形状: {features.shape}")
print(f"工程后特征统计:")
print(f"  总体std范围: {features.std().min():.6f} - {features.std().max():.6f}")
print(f"  特征std中位数: {features.std().median():.6f}")

target = train_df["forward_returns"]

# 训练模型
print("\n=== 训练模型 ===")
model = HullModel(model_type="lightgbm", model_params={"n_estimators": 100})
model.fit(features, target)

# 生成预测并分析
print("\n=== 预测分析 (无标准化) ===")
predictions_raw = model.predict(features, clip=False)
predictions_clipped = model.predict(features, clip=True)

print(f"原始预测统计:")
print(f"  min={predictions_raw.min():.6f}, max={predictions_raw.max():.6f}")
print(f"  mean={predictions_raw.mean():.6f}, std={predictions_raw.std():.6f}")
print(f"  第25百分位={np.percentile(predictions_raw, 25):.6f}")
print(f"  第75百分位={np.percentile(predictions_raw, 75):.6f}")

print(f"\n裁剪后预测统计:")
print(f"  min={predictions_clipped.min():.6f}, max={predictions_clipped.max():.6f}")
print(f"  mean={predictions_clipped.mean():.6f}, std={predictions_clipped.std():.6f}")

# 与标准化版本对比
print(f"\n=== 对比标准化版本 ===")
pipeline_std = FeaturePipeline(standardize=True)
features_std = pipeline_std.fit_transform(train_df)
model_std = HullModel(model_type="lightgbm", model_params={"n_estimators": 100})
model_std.fit(features_std, target)
preds_std = model_std.predict(features_std, clip=False)
print(f"标准化版本预测统计:")
print(f"  min={preds_std.min():.6f}, max={preds_std.max():.6f}")
print(f"  mean={preds_std.mean():.6f}, std={preds_std.std():.6f}")

# 检查特征分布差异
print(f"\n=== 特征分布对比 ===")
print(f"无标准化 - 特征std范围: {features.std().min():.6f} - {features.std().max():.6f}")
print(f"标准化 - 特征std范围: {features_std.std().min():.6f} - {features_std.std().max():.6f}")

# 计算相关性对比
corr_no_std = features.corrwith(target).abs().max()
corr_std = features_std.corrwith(target).abs().max()
print(f"\n与目标最大相关性对比:")
print(f"  无标准化: {corr_no_std:.4f}")
print(f"  标准化: {corr_std:.4f}")

# 查看一些具体特征的差异
print(f"\n=== 重要特征对比 (前5个) ===")
important_features = ['lagged_forward_returns', 'lagged_market_forward_excess_returns', 'M4', 'V13', 'dummy_row_mean']
for feat in important_features:
    if feat in features.columns:
        no_std_corr = abs(features[feat].corr(target))
        std_corr = abs(features_std[feat].corr(target))
        print(f"{feat}:")
        print(f"  无标准化相关性: {no_std_corr:.4f}")
        print(f"  标准化相关性: {std_corr:.4f}")
        print(f"  无标准化std: {features[feat].std():.6f}")
        print(f"  标准化std: {features_std[feat].std():.6f}")
        print()
