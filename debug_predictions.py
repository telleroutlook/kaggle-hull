#!/usr/bin/env python3
"""Debug script to analyze prediction distribution issues"""

import sys
import os
sys.path.insert(0, '/home/dev/github/kaggle-hull/working')

import pandas as pd
import numpy as np
from lib.data import load_train_data
from lib.features import FeaturePipeline
from lib.models import HullModel

# 加载一个小样本数据
print("=== 加载训练数据 ===")
train_df = load_train_data()
print(f"训练数据形状: {train_df.shape}")
print(f"数据列数: {len(train_df.columns)}")

# 检查是否有lagged特征
lagged_cols = [col for col in train_df.columns if col.startswith('lagged_')]
print(f"Lagged特征: {lagged_cols}")

# 特征工程
print("\n=== 特征工程 ===")
pipeline = FeaturePipeline()
features = pipeline.fit_transform(train_df)
print(f"特征矩阵形状: {features.shape}")
print(f"特征数量: {len(features.columns)}")

# 目标变量
target = train_df["forward_returns"]
print(f"目标变量统计: min={target.min():.4f}, max={target.max():.4f}, std={target.std():.4f}")

# 训练一个小模型
print("\n=== 训练模型 ===")
model = HullModel(model_type="lightgbm", model_params={"n_estimators": 100})
model.fit(features, target)

# 生成预测并分析
print("\n=== 预测分析 ===")
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

# 检查特征是否有效
print(f"\n=== 特征质量检查 ===")
print(f"特征矩阵缺失值: {features.isnull().sum().sum()}")
print(f"特征矩阵无穷值: {np.isinf(features.values).sum()}")
print(f"特征矩阵标准差为0的列数: {(features.std() == 0).sum()}")

# 显示一些特征样本
print(f"\n=== 特征样本 (前5行) ===")
print(features.head())

print(f"\n=== 目标变量分布 ===")
print(f"目标变量前10个值: {target.head(10).values}")
print(f"目标变量统计:")
print(f"  零值比例: {(target == 0).mean():.4f}")
print(f"  正值比例: {(target > 0).mean():.4f}")
print(f"  负值比例: {(target < 0).mean():.4f}")