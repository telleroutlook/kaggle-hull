"""
特征工程工具
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Optional


def engineer_features(df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """基础特征工程"""
    
    if feature_cols is None:
        from .data import get_feature_columns
        feature_cols = get_feature_columns(df)
    
    # 创建特征副本
    features = df[feature_cols].copy()
    
    # 处理缺失值
    features = handle_missing_values(features)
    
    # 添加基础统计特征
    features = add_statistical_features(features)
    
    return features


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """处理缺失值"""
    
    # 对于数值特征，使用中位数填充
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # 对于分类特征，使用众数填充
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else "unknown")
    
    return df


def add_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加统计特征"""
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # 添加滚动统计特征
    for window in [5, 10, 20]:
        for col in numeric_cols:
            if len(df) >= window:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
    
    # 添加滞后特征
    for lag in [1, 2, 3]:
        for col in numeric_cols:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
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


def get_feature_groups() -> dict:
    
    return {
        'market': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18'],
        'economic': ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12', 'E13', 'E14', 'E15', 'E16', 'E17', 'E18', 'E19', 'E20'],
        'interest': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9'],
        'price': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11', 'P12', 'P13'],
        'volatility': ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13'],
        'sentiment': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12'],
        'momentum': ['MOM1', 'MOM2', 'MOM3', 'MOM4', 'MOM5', 'MOM6', 'MOM7', 'MOM8', 'MOM9'],
        'dummy': ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9']
    }


__all__ = [
    "engineer_features",
    "handle_missing_values", 
    "add_statistical_features",
    "get_feature_groups",
]