"""
特征工程工具
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Iterable, List, Optional

from .data import get_feature_columns as data_get_feature_columns


def engineer_features(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    *,
    pipeline: FeaturePipeline | None = None,
    return_pipeline: bool = False,
    pipeline_kwargs: Optional[Dict[str, object]] = None,
) -> pd.DataFrame | tuple[pd.DataFrame, FeaturePipeline]:
    """简化版特征工程入口，对齐 FeaturePipeline 逻辑。

    Args:
        df: 原始数据，其中至少包含特征列。
        feature_cols: 显式特征列；未指定时会使用 data.get_feature_columns。
        pipeline: 可复用的 FeaturePipeline 实例。
        return_pipeline: 是否在返回值中附带 pipeline，方便调用方复用。
        pipeline_kwargs: 构造新 pipeline 时使用的参数（如 clip_quantile 等）。
    """

    if feature_cols is None:
        feature_cols = data_get_feature_columns(df)

    feature_view = df[feature_cols].copy()
    active_pipeline = pipeline or FeaturePipeline(**(pipeline_kwargs or {}))
    features = (
        active_pipeline.fit_transform(feature_view, feature_cols)
        if pipeline is None
        else active_pipeline.transform(feature_view)
    )

    if return_pipeline:
        return features, active_pipeline
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
    """添加统计特征，使用向量化Rolling减少重复计算"""

    try:
        from .config import get_config

        config = get_config()
        features_config = config.get_features_config()
        max_features = features_config['max_features']
        rolling_windows = features_config['rolling_windows']
        lag_periods = features_config['lag_periods']
    except ImportError:
        max_features = 20
        rolling_windows = [5, 10, 20]
        lag_periods = [1, 2, 3]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return df

    selected_cols = numeric_cols[: min(max_features, len(numeric_cols))]
    selected = df[selected_cols]
    feature_frames: list[pd.DataFrame] = []

    for window in rolling_windows:
        if len(df) < window:
            continue
        rolling_window = selected.rolling(window=window, min_periods=window)
        means = rolling_window.mean().shift(1).add_suffix(f"_rolling_mean_{window}")
        stds = (
            rolling_window.std(ddof=0)
            .fillna(0.0)
            .shift(1)
            .add_suffix(f"_rolling_std_{window}")
        )
        feature_frames.extend([means.astype("float32"), stds.astype("float32")])

    lag_frames = {
        f"{col}_lag_{lag}": selected[col].shift(lag).astype("float32")
        for lag in lag_periods
        for col in selected_cols
    }
    if lag_frames:
        feature_frames.append(pd.DataFrame(lag_frames, index=df.index))

    if not feature_frames:
        return df

    enriched = pd.concat([df] + feature_frames, axis=1)
    return enriched


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
        self._cached_output_columns: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame, feature_cols: Optional[Iterable[str]] = None) -> "FeaturePipeline":
        """Collect statistics needed for deterministic transforms."""

        if feature_cols is None:
            feature_cols = get_feature_columns(df)

        self.feature_columns = list(feature_cols)
        self._cached_output_columns = None
        features = df[self.feature_columns].copy()
        self.numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = [col for col in self.feature_columns if col not in self.numeric_columns]

        # Fill values for consistent inference.
        for col in self.numeric_columns:
            series = pd.to_numeric(features[col], errors="coerce")
            median = float(series.median(skipna=True)) if series.notnull().any() else 0.0
            fill_value = median
            self.fill_values[col] = fill_value

            if self.clip_quantile > 0 and series.notnull().sum() > 10:
                lower = float(series.quantile(self.clip_quantile))
                upper = float(series.quantile(1 - self.clip_quantile))
            else:
                min_val = series.min(skipna=True)
                max_val = series.max(skipna=True)
                lower = float(min_val) if pd.notna(min_val) else float(fill_value)
                upper = float(max_val) if pd.notna(max_val) else float(fill_value)
            if not np.isfinite(lower):
                lower = float(fill_value)
            if not np.isfinite(upper):
                upper = float(fill_value)
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
        if self._cached_output_columns is None:
            extra_cols: List[str] = []
            extra_cols.extend(f"{col}_is_missing" for col in self.indicator_columns)
            if self.extra_group_stats:
                for group_name, cols in self.group_features.items():
                    if not cols:
                        continue
                    extra_cols.append(f"{group_name}_row_mean")
                    extra_cols.append(f"{group_name}_row_std")
            self._cached_output_columns = self.feature_columns + extra_cols
        return self._cached_output_columns


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


get_feature_columns = data_get_feature_columns


__all__ = [
    "engineer_features",
    "handle_missing_values", 
    "add_statistical_features",
    "FeaturePipeline",
    "get_feature_groups",
    "get_feature_columns",
]
