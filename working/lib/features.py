"""
特征工程工具
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

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
    active_pipeline = pipeline or build_feature_pipeline(**(pipeline_kwargs or {}))
    features = (
        active_pipeline.fit_transform(feature_view, feature_cols)
        if pipeline is None
        else active_pipeline.transform(feature_view)
    )

    if return_pipeline:
        return features, active_pipeline
    return features


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """处理缺失值和无穷值"""
    
    # 对于数值特征，使用中位数填充
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # 替换无穷值为NaN，然后进行填充
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        # 使用中位数填充
        df[col] = df[col].fillna(df[col].median())
        # 如果中位数也是NaN，使用0填充
        df[col] = df[col].fillna(0.0)
    
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

    _CONFIG_FIELDS = (
        "clip_quantile",
        "missing_indicator_threshold",
        "standardize",
        "dtype",
        "extra_group_stats",
        "enable_feature_selection",
        "max_features",
        "stateful",
        "stateful_max_history",
    )

    def __init__(
        self,
        *,
        clip_quantile: float = 0.01,
        missing_indicator_threshold: float = 0.05,
        standardize: bool = False,
        dtype: str = "float32",
        extra_group_stats: bool = True,
        enable_feature_selection: bool = False,
        max_features: int = 300,
        stateful: bool = False,
        stateful_max_history: int = 256,
    ) -> None:
        self.clip_quantile = clip_quantile
        self.missing_indicator_threshold = missing_indicator_threshold
        self.standardize = standardize
        self.dtype = dtype
        self.extra_group_stats = extra_group_stats
        self.enable_feature_selection = enable_feature_selection
        self.max_features = max_features
        self.stateful = stateful
        self.stateful_max_history = max(1, int(stateful_max_history))

        self.feature_columns: List[str] = []
        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.fill_values: Dict[str, object] = {}
        self.clip_bounds: Dict[str, tuple[float, float]] = {}
        self.standardization_stats: Dict[str, tuple[float, float]] = {}
        self.indicator_columns: List[str] = []
        self.group_features: Dict[str, List[str]] = {}
        self._cached_output_columns: Optional[List[str]] = None
        self.selected_features: Optional[List[str]] = None
        self.feature_importance: Optional[Dict[str, float]] = None
        self._history_buffer: Optional[pd.DataFrame] = None

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

        # 特征选择
        if self.enable_feature_selection and self.feature_importance:
            # 根据重要性排序选择特征
            sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            self.selected_features = [feat for feat, _ in sorted_features[:self.max_features]]
        
        # Reset any streaming state because fit has been rerun.
        self._history_buffer = None

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply stored statistics to new data."""

        if not self.feature_columns:
            raise RuntimeError("FeaturePipeline must be fit before calling transform.")

        new_rows = len(df)
        new_index = df.index
        history_enabled = self.stateful
        augmented_df = df
        if history_enabled:
            history = self._history_buffer
            if history is not None and not history.empty:
                augmented_df = pd.concat([history, df], axis=0, copy=False)
            else:
                augmented_df = df.copy()
        df = augmented_df

        features = df.reindex(columns=self.feature_columns, fill_value=np.nan).copy()

        for col in self.numeric_columns:
            series = pd.to_numeric(features[col], errors="coerce")
            # Replace inf values with NaN first
            series = series.replace([np.inf, -np.inf], np.nan)
            fill_value = self.fill_values.get(col, 0.0)
            series = series.fillna(fill_value)
            
            # Get safe clip bounds
            if self.clip_bounds.get(col):
                lower, upper = self.clip_bounds[col]
            else:
                # Use safe bounds that exclude inf values
                valid_series = series[np.isfinite(series)] if len(series[np.isfinite(series)]) > 0 else series
                if len(valid_series) > 0:
                    lower, upper = valid_series.min(), valid_series.max()
                else:
                    lower, upper = -10.0, 10.0
            
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

        # 添加增强特征工程
        features = self._add_enhanced_features(features, df)
        features = self._add_lagged_interactions(features, df)

        # 特征选择
        if self.enable_feature_selection and self.selected_features:
            # 只保留选定的特征
            features = features[self.selected_features]

        features = features.fillna(0)

        if history_enabled:
            if new_rows:
                features = features.iloc[-new_rows:, :].copy()
                features.index = new_index
            else:
                features = features.iloc[0:0].copy()
            self._history_buffer = df.iloc[-self.stateful_max_history :, :].copy()

        return features

    def _add_enhanced_features(self, features: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
        """添加增强特征工程技术指标和历史信号"""
        
        enhanced_frames = []
        
        # 1. 滞后特征交互
        lagged_features = [col for col in original_df.columns if col.startswith('lagged_')]
        for lag_col in lagged_features:
            if lag_col in original_df.columns:
                # 与主要市场特征交互
                for market_col in ['M1', 'M2', 'M3', 'M4', 'M5', 'P1', 'P2', 'P3', 'V1', 'V2']:
                    if market_col in features.columns:
                        interaction_name = f"{lag_col}_x_{market_col}"
                        interaction = features[market_col] * original_df[lag_col]
                        interaction = interaction.fillna(0).astype(self.dtype)
                        enhanced_frames.append(interaction.to_frame(interaction_name))
        
        # 2. 动量增强特征
        if 'MOM1' in features.columns and 'M1' in original_df.columns:
            momentum_enhanced = features['MOM1'] * original_df['M1']
            enhanced_frames.append(momentum_enhanced.fillna(0).astype(self.dtype).to_frame('MOM1_M1_enhanced'))
        
        # 3. 波动率状态特征
        if 'V1' in features.columns:
            vol_zscore = (features['V1'] - features['V1'].rolling(20).mean().fillna(features['V1'])) / features['V1'].rolling(20).std().fillna(1)
            enhanced_frames.append(vol_zscore.fillna(0).astype(self.dtype).to_frame('V1_volatility_state'))
        
        # 4. 技术指标增强
        enhanced_frames.extend(self._add_technical_indicators(original_df, features))
        
        # 5. 交叉特征
        enhanced_frames.extend(self._add_cross_features(features))
        
        if enhanced_frames:
            enhanced_df = pd.concat(enhanced_frames, axis=1)
            features = pd.concat([features, enhanced_df], axis=1)
        
        return features

    def _add_technical_indicators(self, original_df: pd.DataFrame, features: pd.DataFrame) -> List[pd.DataFrame]:
        """添加技术指标特征"""
        
        tech_indicators = []
        
        # RSI (Relative Strength Index)
        if 'P1' in original_df.columns:
            price = original_df['P1']
            delta = price.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            tech_indicators.append(rsi.fillna(50).astype(self.dtype).to_frame('rsi_14'))
        
        # MACD (Moving Average Convergence Divergence)
        if 'P1' in original_df.columns:
            price = original_df['P1']
            exp1 = price.ewm(span=12).mean()
            exp2 = price.ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            histogram = macd - signal
            tech_indicators.append(macd.fillna(0).astype(self.dtype).to_frame('macd_line'))
            tech_indicators.append(signal.fillna(0).astype(self.dtype).to_frame('macd_signal'))
            tech_indicators.append(histogram.fillna(0).astype(self.dtype).to_frame('macd_histogram'))
        
        # 移动平均交叉
        if 'P1' in original_df.columns:
            price = original_df['P1']
            ma_5 = price.rolling(5).mean()
            ma_20 = price.rolling(20).mean()
            ma_cross = ma_5 / ma_20
            tech_indicators.append(ma_cross.fillna(1).astype(self.dtype).to_frame('ma_cross_ratio'))
        
        # 布林带
        if 'P1' in original_df.columns:
            price = original_df['P1']
            ma_20 = price.rolling(20).mean()
            std_20 = price.rolling(20).std()
            bb_upper = ma_20 + (2 * std_20)
            bb_lower = ma_20 - (2 * std_20)
            bb_position = (price - bb_lower) / (bb_upper - bb_lower)
            bb_width = (bb_upper - bb_lower) / ma_20
            tech_indicators.append(bb_position.fillna(0.5).astype(self.dtype).to_frame('bollinger_position'))
            tech_indicators.append(bb_width.fillna(0.1).astype(self.dtype).to_frame('bollinger_width'))
        
        return tech_indicators

    def _add_cross_features(self, features: pd.DataFrame) -> List[pd.DataFrame]:
        """添加特征交叉项"""
        
        cross_features = []
        
        # 重要的特征交叉
        important_crosses = [
            ('M1', 'M2', 'market_corr'),
            ('P1', 'V1', 'price_vol_interaction'),
            ('E1', 'E2', 'economic_dual'),
            ('MOM1', 'M1', 'momentum_market_interaction'),
            ('V1', 'P1', 'vol_price_interaction'),
            ('MOM1', 'V1', 'momentum_vol_interaction'),
        ]
        
        for col1, col2, name in important_crosses:
            if col1 in features.columns and col2 in features.columns:
                cross_product = features[col1] * features[col2]
                cross_features.append(cross_product.fillna(0).astype(self.dtype).to_frame(name))
        
        # 比率特征
        ratio_features = [
            ('P1', 'P2', 'price_ratio'),
            ('V1', 'V2', 'vol_ratio'),
            ('M1', 'M2', 'market_ratio'),
            ('E1', 'E2', 'economic_ratio'),
            ('MOM1', 'MOM2', 'momentum_ratio'),
        ]
        
        for col1, col2, name in ratio_features:
            if col1 in features.columns and col2 in features.columns:
                ratio = features[col1] / (features[col2] + 1e-8)
                ratio = ratio.replace([np.inf, -np.inf], 0).fillna(1)
                cross_features.append(ratio.astype(self.dtype).to_frame(name))
        
        # 统计特征交叉
        if 'M1' in features.columns and 'V1' in features.columns:
            # 市场特征与波动率的标准化交叉
            norm_cross = (features['M1'] * features['V1']) / (features['V1'].std() + 1e-8)
            cross_features.append(norm_cross.fillna(0).astype(self.dtype).to_frame('norm_market_vol_cross'))
        
        return cross_features

    def _add_lagged_interactions(self, features: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
        """添加滞后特征的高级交互"""
        
        lagged_cols = sorted(col for col in original_df.columns if col.startswith('lagged_'))
        
        if not lagged_cols:
            return features
        
        interaction_frames = []
        
        # 滞后特征与主要特征的多阶交互
        for lag_col in lagged_cols:
            if lag_col not in original_df.columns:
                continue
                
            # 滞后特征的变化率
            lag_change = original_df[lag_col].pct_change().fillna(0)
            interaction_frames.append(lag_change.astype(self.dtype).to_frame(f"{lag_col}_change_rate"))
            
            # 滞后特征与主要市场特征的时间交互
            for market_col in ['M1', 'M2', 'M3', 'V1', 'V2']:
                if market_col in features.columns:
                    time_interaction = features[market_col] * original_df[lag_col].shift(1)
                    interaction_frames.append(time_interaction.fillna(0).astype(self.dtype).to_frame(f"{lag_col}_x_{market_col}_lag1"))
        
        if interaction_frames:
            interaction_df = pd.concat(interaction_frames, axis=1)
            features = pd.concat([features, interaction_df], axis=1)
        
        return features

    def fit_transform(self, df: pd.DataFrame, feature_cols: Optional[Iterable[str]] = None) -> pd.DataFrame:
        return self.fit(df, feature_cols).transform(df)

    def set_feature_importance(self, importance_dict: Dict[str, float]) -> None:
        """设置特征重要性用于特征选择"""
        self.feature_importance = importance_dict

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "FeaturePipeline":
        """Construct a pipeline instance from a serialized config mapping."""

        allowed = {field: config[field] for field in cls._CONFIG_FIELDS if field in config}
        return cls(**allowed)

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        """Return the canonical default pipeline configuration."""

        return {
            "clip_quantile": 0.01,
            "missing_indicator_threshold": 0.05,
            "standardize": False,
            "dtype": "float32",
            "extra_group_stats": True,
            "enable_feature_selection": False,
            "max_features": 300,
            "stateful": False,
            "stateful_max_history": 256,
        }

    def to_config(self) -> Dict[str, Any]:
        """Export the current pipeline hyper-parameters to a plain dict."""

        return {field: getattr(self, field) for field in self._CONFIG_FIELDS}

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


def _normalize_for_hash(value: Any) -> Any:
    """Convert numpy/scalar types into JSON-serializable primitives."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_normalize_for_hash(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize_for_hash(v) for k, v in value.items()}
    return value


def build_feature_pipeline(**overrides: Any) -> FeaturePipeline:
    """Create a FeaturePipeline with shared defaults plus optional overrides."""

    config = FeaturePipeline.default_config()
    for key, value in overrides.items():
        if value is None or key not in FeaturePipeline._CONFIG_FIELDS:
            continue
        config[key] = value
    return FeaturePipeline.from_config(config)


def pipeline_config_hash(
    config: Mapping[str, Any],
    *,
    augment_flag: bool | None = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> str:
    """Return a deterministic hash describing the pipeline + data knobs."""

    payload: Dict[str, Any] = {"pipeline": _normalize_for_hash(dict(config))}
    if augment_flag is not None:
        payload["augment_data"] = bool(augment_flag)
    if extra:
        payload["extra"] = _normalize_for_hash(dict(extra))
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


__all__ = [
    "engineer_features",
    "handle_missing_values", 
    "add_statistical_features",
    "FeaturePipeline",
    "get_feature_groups",
    "get_feature_columns",
    "build_feature_pipeline",
    "pipeline_config_hash",
]
