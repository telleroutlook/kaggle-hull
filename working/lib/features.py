"""
特征工程工具
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Iterable, List, Optional


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

    # 为了在训练/推理阶段保持稳定，确保没有缺失值
    features = features.ffill()
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        median_values = features[numeric_cols].median()
        features[numeric_cols] = features[numeric_cols].fillna(median_values)
        if features[numeric_cols].isnull().any().any():
            features[numeric_cols] = features[numeric_cols].fillna(0)
    non_numeric_cols = [col for col in features.columns if col not in numeric_cols]
    for col in non_numeric_cols:
        if features[col].isnull().any():
            mode_series = features[col].mode()
            fill_value = mode_series.iloc[0] if not mode_series.empty else "unknown"
            features[col] = features[col].fillna(fill_value)

    if features.isnull().any().any():
        features = features.fillna(0)
    
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
    
    # 导入配置
    try:
        from .config import get_config
        config = get_config()
        features_config = config.get_features_config()
        max_features = features_config['max_features']
        rolling_windows = features_config['rolling_windows']
        lag_periods = features_config['lag_periods']
    except ImportError:
        # 如果配置模块不可用，使用默认值
        max_features = 20
        rolling_windows = [5, 10, 20]
        lag_periods = [1, 2, 3]
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # 只对数值列添加特征，避免内存爆炸
    selected_cols = numeric_cols[:min(max_features, len(numeric_cols))]  # 限制特征数量
    
    # 使用字典推导式优化内存使用
    new_features = {}
    
    # 添加滚动统计特征
    for window in rolling_windows:
        if len(df) >= window:
            for col in selected_cols:
                rolling_mean = df[col].rolling(window=window, min_periods=window).mean()
                rolling_std = df[col].rolling(window=window, min_periods=window).std()
                # 向后平移一步，避免使用当前行的信息
                new_features[f'{col}_rolling_mean_{window}'] = rolling_mean.shift(1)
                new_features[f'{col}_rolling_std_{window}'] = rolling_std.shift(1)
    
    # 添加滞后特征
    for lag in lag_periods:
        for col in selected_cols:
            new_features[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # 一次性添加所有新特征，避免DataFrame碎片化
    new_features_df = pd.DataFrame(new_features, index=df.index)
    df = pd.concat([df, new_features_df], axis=1)
    
    return df


class FeaturePipeline:
    """Lightweight feature engineering pipeline shared by training/inference."""

    def __init__(
        self,
        *,
        clip_quantile: float = 0.01,
        missing_indicator_threshold: float = 0.05,
        standardize: bool = True,
        dtype: str = "float32",
        extra_group_stats: bool = True,
    ) -> None:
        self.clip_quantile = clip_quantile
        self.missing_indicator_threshold = missing_indicator_threshold
        self.standardize = standardize
        self.dtype = dtype
        self.extra_group_stats = extra_group_stats

        self.feature_columns: List[str] = []
        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.fill_values: Dict[str, object] = {}
        self.clip_bounds: Dict[str, tuple[float, float]] = {}
        self.standardization_stats: Dict[str, tuple[float, float]] = {}
        self.indicator_columns: List[str] = []
        self.group_features: Dict[str, List[str]] = {}

    def fit(self, df: pd.DataFrame, feature_cols: Optional[Iterable[str]] = None) -> "FeaturePipeline":
        """Collect statistics needed for deterministic transforms."""

        if feature_cols is None:
            feature_cols = get_feature_columns(df)

        self.feature_columns = list(feature_cols)
        features = df[self.feature_columns].copy()
        self.numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = [col for col in self.feature_columns if col not in self.numeric_columns]

        # Fill values for consistent inference.
        for col in self.numeric_columns:
            series = pd.to_numeric(features[col], errors="coerce")
            median = float(series.median(skipna=True)) if series.notnull().any() else 0.0
            self.fill_values[col] = median

            if self.clip_quantile > 0 and series.notnull().sum() > 10:
                lower = float(series.quantile(self.clip_quantile))
                upper = float(series.quantile(1 - self.clip_quantile))
            else:
                min_val = series.min(skipna=True)
                max_val = series.max(skipna=True)
                lower = float(min_val) if pd.notna(min_val) else float(fill_value)
                upper = float(max_val) if pd.notna(max_val) else float(fill_value)
            if lower == upper:
                upper = lower + 1e-6
            self.clip_bounds[col] = (lower, upper)

            std = float(series.std(skipna=True))
            self.standardization_stats[col] = (float(series.mean(skipna=True)), std if std > 0 else 1.0)

        for col in self.categorical_columns:
            modes = features[col].mode(dropna=True)
            self.fill_values[col] = modes.iloc[0] if not modes.empty else "missing"

        # Missing indicators for high-null columns.
        null_rates = features.isnull().mean()
        self.indicator_columns = [
            col
            for col, rate in null_rates.items()
            if rate >= self.missing_indicator_threshold and col in self.feature_columns
        ]

        if self.extra_group_stats:
            groups = get_feature_groups()
            # Only keep valid columns in each group.
            self.group_features = {
                name: [col for col in cols if col in self.feature_columns]
                for name, cols in groups.items()
            }

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply stored statistics to new data."""

        if not self.feature_columns:
            raise RuntimeError("FeaturePipeline must be fit before calling transform.")

        features = df.reindex(columns=self.feature_columns, fill_value=np.nan).copy()

        for col in self.numeric_columns:
            series = pd.to_numeric(features[col], errors="coerce")
            fill_value = self.fill_values.get(col, 0.0)
            series = series.fillna(fill_value)
            lower, upper = self.clip_bounds.get(col, (series.min(), series.max()))
            series = series.clip(lower=lower, upper=upper)
            if self.standardize:
                mean, std = self.standardization_stats.get(col, (0.0, 1.0))
                series = (series - mean) / (std if std > 0 else 1.0)
            features[col] = series.astype(self.dtype)

        for col in self.categorical_columns:
            fill_value = self.fill_values.get(col, "missing")
            features[col] = features[col].fillna(fill_value)

        if self.indicator_columns:
            indicator_data = {}
            for col in self.indicator_columns:
                source = df[col] if col in df.columns else pd.Series(np.nan, index=df.index)
                indicator_data[f"{col}_is_missing"] = source.isnull().astype("int8")
            indicator_df = pd.DataFrame(indicator_data, index=features.index)
            features = pd.concat([features, indicator_df], axis=1)

        if self.extra_group_stats and self.group_features:
            group_frames = {}
            for group_name, cols in self.group_features.items():
                if not cols:
                    continue
                data = features[cols]
                group_frames[f"{group_name}_row_mean"] = data.mean(axis=1).astype(self.dtype)
                group_frames[f"{group_name}_row_std"] = data.std(axis=1).fillna(0).astype(self.dtype)
            if group_frames:
                group_df = pd.DataFrame(group_frames, index=features.index)
                features = pd.concat([features, group_df], axis=1)

        features = features.fillna(0)
        return features

    def fit_transform(self, df: pd.DataFrame, feature_cols: Optional[Iterable[str]] = None) -> pd.DataFrame:
        return self.fit(df, feature_cols).transform(df)

    @property
    def output_columns(self) -> List[str]:
        if not self.feature_columns:
            raise RuntimeError("FeaturePipeline must be fit before accessing output columns.")
        extra_cols = []
        extra_cols.extend(f"{col}_is_missing" for col in self.indicator_columns)
        if self.extra_group_stats:
            for group_name, cols in self.group_features.items():
                if not cols:
                    continue
                extra_cols.append(f"{group_name}_row_mean")
                extra_cols.append(f"{group_name}_row_std")
        return self.feature_columns + extra_cols


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
    "FeaturePipeline",
    "get_feature_groups",
]
