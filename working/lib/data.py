"""
数据加载和预处理工具
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from .env import DataPaths, detect_run_environment, get_data_paths


def load_train_data(data_paths: Optional[DataPaths] = None) -> pd.DataFrame:
    """加载训练数据"""
    
    if data_paths is None:
        env = detect_run_environment()
        data_paths = get_data_paths(env)
    
    print(f"加载训练数据: {data_paths.train_data}")
    df = pd.read_csv(data_paths.train_data)
    print(f"训练数据形状: {df.shape}")
    print(f"训练数据列: {df.columns.tolist()}")
    
    return df


def load_test_data(data_paths: Optional[DataPaths] = None) -> pd.DataFrame:
    """加载测试数据"""
    
    if data_paths is None:
        env = detect_run_environment()
        data_paths = get_data_paths(env)
    
    print(f"加载测试数据: {data_paths.test_data}")
    df = pd.read_csv(data_paths.test_data)
    print(f"测试数据形状: {df.shape}")
    
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """获取特征列名"""
    
    # 排除目标变量和其他非特征列
    exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 
                   'market_forward_excess_returns', 'is_scored',
                   'lagged_forward_returns', 'lagged_risk_free_rate', 
                   'lagged_market_forward_excess_returns']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


def get_target_columns() -> list:
    """获取目标变量列名"""
    
    return ['forward_returns', 'risk_free_rate', 'market_forward_excess_returns']


def validate_data(df: pd.DataFrame, data_type: str = "train") -> bool:
    """验证数据完整性"""
    
    required_cols = ['date_id']
    
    if data_type == "train":
        required_cols.extend(['forward_returns', 'risk_free_rate', 'market_forward_excess_returns'])
    elif data_type == "test":
        required_cols.extend(['is_scored'])
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"警告: 缺少必要的列: {missing_cols}")
        return False
    
    print(f"✅ {data_type}数据验证通过")
    return True


__all__ = [
    "load_train_data",
    "load_test_data", 
    "get_feature_columns",
    "get_target_columns",
    "validate_data",
]