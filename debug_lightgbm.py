#!/usr/bin/env python3
"""Debug script to analyze LightGBM model behavior"""

import sys
import os
sys.path.insert(0, '/home/dev/github/kaggle-hull/working')

import pandas as pd
import numpy as np
from lib.data import load_train_data
from lib.features import FeaturePipeline
from lib.models import HullModel
from lightgbm import LGBMRegressor

# 加载数据
print("=== 加载数据 ===")
train_df = load_train_data()
pipeline = FeaturePipeline()
features = pipeline.fit_transform(train_df)
target = train_df["forward_returns"]

print(f"特征形状: {features.shape}")
print(f"目标形状: {target.shape}")

# 测试默认LightGBM参数
print("\n=== 测试默认LightGBM参数 ===")
model_params = {
    "objective": "regression",
    "n_estimators": 100,
    "learning_rate": 0.005,
    "num_leaves": 256,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 5.0,
    "min_child_samples": 40,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 42
}

model = LGBMRegressor(**model_params)
model.fit(features, target)
preds_default = model.predict(features)
print(f"默认参数预测统计: min={preds_default.min():.6f}, max={preds_default.max():.6f}, std={preds_default.std():.6f}")

# 测试更激进的参数
print("\n=== 测试更激进的参数 ===")
aggressive_params = {
    "objective": "regression",
    "n_estimators": 1000,
    "learning_rate": 0.01,
    "num_leaves": 32,
    "max_depth": 6,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "min_child_samples": 5,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 42
}

model_agg = LGBMRegressor(**aggressive_params)
model_agg.fit(features, target)
preds_agg = model_agg.predict(features)
print(f"激进参数预测统计: min={preds_agg.min():.6f}, max={preds_agg.max():.6f}, std={preds_agg.std():.6f}")

# 测试回归参数
print("\n=== 测试回归参数 ===")
regression_params = {
    "objective": "regression",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": 10,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "min_child_samples": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 42
}

model_reg = LGBMRegressor(**regression_params)
model_reg.fit(features, target)
preds_reg = model_reg.predict(features)
print(f"回归参数预测统计: min={preds_reg.min():.6f}, max={preds_reg.max():.6f}, std={preds_reg.std():.6f}")

# 测试随机森林作为对比
print("\n=== 测试随机森林 ===")
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(features, target)
preds_rf = rf_model.predict(features)
print(f"随机森林预测统计: min={preds_rf.min():.6f}, max={preds_rf.max():.6f}, std={preds_rf.std():.6f}")

# 检查特征重要性
print("\n=== 特征重要性分析 ===")
importance = model.feature_importances_
top_features = np.argsort(importance)[-10:]
print("前10个重要特征:")
for i, feat_idx in enumerate(reversed(top_features)):
    print(f"  {features.columns[feat_idx]}: {importance[feat_idx]:.4f}")

# 检查lagged特征的重要性
lagged_importance = [(col, importance[i]) for i, col in enumerate(features.columns) if 'lagged' in col]
lagged_importance.sort(key=lambda x: x[1], reverse=True)
print("\nLagged特征重要性:")
for col, imp in lagged_importance:
    print(f"  {col}: {imp:.4f}")

print(f"\n=== 目标变量分析 ===")
print(f"目标变量均值: {target.mean():.6f}")
print(f"目标变量标准差: {target.std():.6f}")
print(f"预测值均值: {preds_default.mean():.6f}")
print(f"预测值与目标相关系数: {np.corrcoef(preds_default, target)[0,1]:.4f}")