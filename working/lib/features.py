"""增强特征工程工具

包含高级技术指标、统计特征、数据质量改进和特征稳定性分析
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from collections import defaultdict
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE, SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import mutual_info_score
from sklearn.cluster import KMeans
from scipy.stats import spearmanr, pearsonr, kendalltau

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
    """Enhanced feature engineering pipeline with intelligent feature selection and optimization."""

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
        "enable_data_quality",
        "enable_feature_stability",
        "outlier_detection",
        "missing_value_strategy",
        "enable_intelligent_selection",
        "enable_feature_combinations",
        "enable_tiered_features",
        "enable_robust_scaling",
        "feature_selection_method",
        "combination_complexity",
        "tiered_levels",
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
        enable_data_quality: bool = True,
        enable_feature_stability: bool = True,
        outlier_detection: bool = True,
        missing_value_strategy: str = "median",
        enable_intelligent_selection: bool = True,
        enable_feature_combinations: bool = True,
        enable_tiered_features: bool = True,
        enable_robust_scaling: bool = True,
        feature_selection_method: str = "mixed",  # "correlation", "mutual_info", "rfe", "mixed"
        combination_complexity: int = 3,  # 1-5, complexity level for feature combinations
        tiered_levels: int = 4,  # number of market state tiers
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
        self.enable_data_quality = enable_data_quality
        self.enable_feature_stability = enable_feature_stability
        self.outlier_detection = outlier_detection
        self.missing_value_strategy = missing_value_strategy
        
        # 智能特征工程配置
        self.enable_intelligent_selection = enable_intelligent_selection
        self.enable_feature_combinations = enable_feature_combinations
        self.enable_tiered_features = enable_tiered_features
        self.enable_robust_scaling = enable_robust_scaling
        self.feature_selection_method = feature_selection_method
        self.combination_complexity = combination_complexity
        self.tiered_levels = tiered_levels

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
        
        # 数据质量和特征稳定性属性
        self.feature_stability_scores: Optional[Dict[str, float]] = None
        self.outlier_bounds: Dict[str, Tuple[float, float]] = {}
        self.data_quality_metrics: Dict[str, Any] = {}
        self.scaler: Optional[Any] = None
        
        # 智能特征工程属性
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.mutual_info_scores: Optional[Dict[str, float]] = None
        self.feature_compatibility: Optional[Dict[str, List[str]]] = None
        self.market_state_tiers: Optional[Dict[str, pd.Series]] = None
        self.selected_features_meta: Optional[Dict[str, Any]] = None
        self.combination_history: List[Dict[str, Any]] = []

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

        # 数据质量分析
        if self.enable_data_quality:
            self._analyze_data_quality(features)
        
        # 特征稳定性分析
        if self.enable_feature_stability:
            self._analyze_feature_stability(features)
        
        # 异常值检测和边界设置
        if self.outlier_detection:
            self._set_outlier_bounds(features)
        
        # 初始化智能缩放器
        if self.standardize and self.enable_robust_scaling:
            self._initialize_scaler(features)
        
        # 智能特征选择
        if self.enable_intelligent_selection:
            self._perform_intelligent_feature_selection(features)
        elif self.enable_feature_selection and self.feature_importance:
            # 根据重要性排序选择特征
            sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            self.selected_features = [feat for feat, _ in sorted_features[:self.max_features]]
        
        # 分析特征相关性
        if self.enable_intelligent_selection:
            self._analyze_feature_correlations(features)
        
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
            
            # 智能缺失值填充
            series = self._smart_fill_missing(series, col)
            
            # 异常值处理
            if self.outlier_detection and col in self.outlier_bounds:
                lower, upper = self.outlier_bounds[col]
                series = series.clip(lower=lower, upper=upper)
            elif self.clip_bounds.get(col):
                lower, upper = self.clip_bounds[col]
                series = series.clip(lower=lower, upper=upper)
            else:
                # Use safe bounds that exclude inf values
                valid_series = series[np.isfinite(series)] if len(series[np.isfinite(series)]) > 0 else series
                if len(valid_series) > 0:
                    lower, upper = valid_series.min(), valid_series.max()
                else:
                    lower, upper = -10.0, 10.0
                series = series.clip(lower=lower, upper=upper)
            
            if self.standardize:
                if self.enable_robust_scaling and self.scaler is not None:
                    # 使用RobustScaler或QuantileTransformer
                    try:
                        if hasattr(self.scaler, 'transform'):
                            # 保持DataFrame格式以保留特征名称，避免QuantileTransformer警告
                            if isinstance(self.scaler, QuantileTransformer):
                                # 对于QuantileTransformer，使用DataFrame保持特征名称
                                series_df = pd.DataFrame({col: series.values})
                                scaled_values = self.scaler.transform(series_df)
                                series = pd.Series(scaled_values.flatten(), index=series.index, name=col)
                            else:
                                # 对于其他scaler，可以安全地使用numpy数组
                                series_values = series.values.reshape(-1, 1)
                                scaled_values = self.scaler.transform(series_values)
                                series = pd.Series(scaled_values.flatten(), index=series.index)
                        else:
                            mean, std = self.standardization_stats.get(col, (0.0, 1.0))
                            series = (series - mean) / (std if std > 0 else 1.0)
                    except Exception:
                        # 回退到StandardScaler
                        mean, std = self.standardization_stats.get(col, (0.0, 1.0))
                        series = (series - mean) / (std if std > 0 else 1.0)
                else:
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
        
        # 智能特征组合
        if self.enable_feature_combinations:
            features = self._add_intelligent_combinations(features, df)
        
        # 分层特征工程
        if self.enable_tiered_features:
            features = self._add_tiered_market_features(features, df)

        # 智能特征选择
        if self.enable_intelligent_selection and self.selected_features:
            features = features[self.selected_features]
        elif self.enable_feature_selection and self.selected_features:
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
        
        # 5. 分层统计特征
        enhanced_frames.extend(self._add_tiered_statistics(original_df, features))
        
        # 6. 宏观因子交互
        enhanced_frames.extend(self._add_macro_factor_interactions(original_df, features))
        
        # 5. 交叉特征
        enhanced_frames.extend(self._add_cross_features(features))
        
        if enhanced_frames:
            enhanced_df = pd.concat(enhanced_frames, axis=1)
            features = pd.concat([features, enhanced_df], axis=1)
        
        return features

    def _add_technical_indicators(self, original_df: pd.DataFrame, features: pd.DataFrame) -> List[pd.DataFrame]:
        """添加高级技术指标特征"""
        
        tech_indicators = []
        
        # RSI (Relative Strength Index) with multiple periods
        if 'P1' in original_df.columns:
            price = original_df['P1']
            for period in [7, 14, 21]:
                delta = price.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                tech_indicators.append(rsi.fillna(50).astype(self.dtype).to_frame(f'rsi_{period}'))
        
        # Williams %R
        if 'P1' in original_df.columns:
            price = original_df['P1']
            for period in [14, 21]:
                high = price.rolling(window=period).max()
                low = price.rolling(window=period).min()
                williams_r = -100 * ((high - price) / (high - low + 1e-8))
                tech_indicators.append(williams_r.fillna(-50).astype(self.dtype).to_frame(f'williams_r_{period}'))
        
        # Stochastic Oscillator
        if 'P1' in original_df.columns:
            price = original_df['P1']
            for period in [14, 21]:
                high = price.rolling(window=period).max()
                low = price.rolling(window=period).min()
                k_percent = 100 * ((price - low) / (high - low + 1e-8))
                k_percent_ma = k_percent.rolling(window=3).mean()
                d_percent = k_percent_ma.rolling(window=3).mean()
                tech_indicators.append(k_percent.fillna(50).astype(self.dtype).to_frame(f'stoch_k_{period}'))
                tech_indicators.append(d_percent.fillna(50).astype(self.dtype).to_frame(f'stoch_d_{period}'))
        
        # ADX (Average Directional Index)
        if 'P1' in original_df.columns:
            price = original_df['P1']
            high = price.rolling(window=14).max()
            low = price.rolling(window=14).min()
            
            # True Range calculation
            prev_close = price.shift(1)
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Directional Movement
            up_move = price - price.shift(1)
            down_move = price.shift(1) - price
            
            plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
            minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
            
            # Smooth the indicators
            atr = tr.rolling(window=14).mean()
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
            
            # ADX calculation
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
            adx = dx.rolling(window=14).mean()
            
            tech_indicators.append(plus_di.fillna(0).astype(self.dtype).to_frame('adx_plus_di'))
            tech_indicators.append(minus_di.fillna(0).astype(self.dtype).to_frame('adx_minus_di'))
            tech_indicators.append(adx.fillna(0).astype(self.dtype).to_frame('adx_value'))
        
        # MACD (Moving Average Convergence Divergence) with multiple periods
        if 'P1' in original_df.columns:
            price = original_df['P1']
            for (fast, slow, signal) in [(12, 26, 9), (5, 35, 5)]:
                exp1 = price.ewm(span=fast).mean()
                exp2 = price.ewm(span=slow).mean()
                macd = exp1 - exp2
                signal_line = macd.ewm(span=signal).mean()
                histogram = macd - signal_line
                tech_indicators.append(macd.fillna(0).astype(self.dtype).to_frame(f'macd_{fast}_{slow}'))
                tech_indicators.append(signal_line.fillna(0).astype(self.dtype).to_frame(f'macd_signal_{fast}_{slow}'))
                tech_indicators.append(histogram.fillna(0).astype(self.dtype).to_frame(f'macd_hist_{fast}_{slow}'))
        
        # 移动平均交叉 (多时间框架)
        if 'P1' in original_df.columns:
            price = original_df['P1']
            ma_pairs = [(5, 10), (5, 20), (10, 20), (20, 50)]
            for short, long in ma_pairs:
                ma_short = price.rolling(short).mean()
                ma_long = price.rolling(long).mean()
                ma_cross = ma_short / ma_long
                tech_indicators.append(ma_cross.fillna(1).astype(self.dtype).to_frame(f'ma_cross_{short}_{long}'))
        
        # 布林带增强版
        if 'P1' in original_df.columns:
            price = original_df['P1']
            for period in [20, 50]:
                ma = price.rolling(period).mean()
                std = price.rolling(period).std()
                bb_upper = ma + (2 * std)
                bb_lower = ma - (2 * std)
                bb_width = (bb_upper - bb_lower) / ma
                bb_squeeze = (std / ma) / (price.rolling(50).std() / price.rolling(50).mean())
                
                tech_indicators.append(bb_width.fillna(0.1).astype(self.dtype).to_frame(f'bollinger_width_{period}'))
                tech_indicators.append(bb_squeeze.fillna(1).astype(self.dtype).to_frame(f'bollinger_squeeze_{period}'))
        
        # 波动率指标
        if 'V1' in original_df.columns:
            vol = original_df['V1']
            for period in [10, 20]:
                vol_mean = vol.rolling(period).mean()
                vol_ratio = vol / vol_mean
                tech_indicators.append(vol_ratio.fillna(1).astype(self.dtype).to_frame(f'vol_ratio_{period}'))
        
        return tech_indicators

    def _add_cross_features(self, features: pd.DataFrame) -> List[pd.DataFrame]:
        """添加特征交叉项和高级组合特征"""
        
        cross_features = []
        
        # 重要的特征交叉
        important_crosses = [
            ('M1', 'M2', 'market_corr'),
            ('P1', 'V1', 'price_vol_interaction'),
            ('E1', 'E2', 'economic_dual'),
            ('MOM1', 'M1', 'momentum_market_interaction'),
            ('V1', 'P1', 'vol_price_interaction'),
            ('MOM1', 'V1', 'momentum_vol_interaction'),
            ('E1', 'V1', 'economic_vol_interaction'),
            ('S1', 'P1', 'sentiment_price_interaction'),
            ('I1', 'M1', 'interest_market_interaction'),
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
            ('S1', 'S2', 'sentiment_ratio'),
        ]
        
        for col1, col2, name in ratio_features:
            if col1 in features.columns and col2 in features.columns:
                ratio = features[col1] / (features[col2] + 1e-8)
                ratio = ratio.replace([np.inf, -np.inf], 0).fillna(1)
                cross_features.append(ratio.astype(self.dtype).to_frame(name))
        
        # 高级交叉特征
        if all(col in features.columns for col in ['M1', 'V1', 'MOM1']):
            # 三元交叉特征
            ternary_cross = features['M1'] * features['V1'] * features['MOM1']
            cross_features.append(ternary_cross.fillna(0).astype(self.dtype).to_frame('ternary_market_vol_mom'))
        
        if all(col in features.columns for col in ['E1', 'I1', 'P1']):
            # 宏观-利率-价格交互
            macro_interaction = features['E1'] * features['I1'] * features['P1']
            cross_features.append(macro_interaction.fillna(0).astype(self.dtype).to_frame('macro_interest_price_interaction'))
        
        # 统计特征交叉
        if 'M1' in features.columns and 'V1' in features.columns:
            # 市场特征与波动率的标准化交叉
            norm_cross = (features['M1'] * features['V1']) / (features['V1'].std() + 1e-8)
            cross_features.append(norm_cross.fillna(0).astype(self.dtype).to_frame('norm_market_vol_cross'))
        
        return cross_features

    def _add_tiered_statistics(self, original_df: pd.DataFrame, features: pd.DataFrame) -> List[pd.DataFrame]:
        """添加分层统计特征 - 根据市场状态分层的统计特征"""
        
        tiered_stats = []
        
        # 定义市场状态基于波动率和趋势
        if 'V1' in features.columns and 'M1' in original_df.columns:
            # 波动率状态
            vol_percentile = features['V1'].rolling(50, min_periods=20).rank(pct=True)
            low_vol = vol_percentile < 0.25
            high_vol = vol_percentile > 0.75
            
            # 趋势状态 (基于价格动量)
            if 'P1' in original_df.columns:
                price_change = original_df['P1'].pct_change()
                trend_strength = price_change.rolling(20, min_periods=10).std()
                strong_trend = trend_strength > trend_strength.rolling(50, min_periods=20).quantile(0.75)
            else:
                strong_trend = pd.Series(False, index=features.index)
            
            # 分层统计：按波动率和趋势组合
            for condition_name, condition in [
                ('low_vol', low_vol),
                ('high_vol', high_vol), 
                ('strong_trend', strong_trend),
                ('normal', ~(low_vol | high_vol | strong_trend))
            ]:
                if condition.sum() < 10:  # 确保有足够的样本
                    continue
                    
                # 分层均值和标准差
                grouped_mean = features.where(condition).rolling(5, min_periods=1).mean()
                grouped_std = features.where(condition).rolling(5, min_periods=1).std()
                
                # 只为数值特征计算
                numeric_features = features.select_dtypes(include=[np.number]).columns
                for col in numeric_features[:20]:  # 限制特征数量
                    if col in grouped_mean.columns:
                        tiered_stats.append(
                            grouped_mean[col].fillna(features[col].mean()).astype(self.dtype).to_frame(f'{col}_mean_{condition_name}')
                        )
                        tiered_stats.append(
                            grouped_std[col].fillna(features[col].std()).astype(self.dtype).to_frame(f'{col}_std_{condition_name}')
                        )
        
        return tiered_stats

    def _add_macro_factor_interactions(self, original_df: pd.DataFrame, features: pd.DataFrame) -> List[pd.DataFrame]:
        """添加宏观因子交互特征"""
        
        macro_interactions = []
        
        # 利率-市场交互
        if all(col in features.columns for col in ['I1', 'M1', 'M2']):
            # 利率调整的市场相关性
            rate_adjusted_market = features['M1'] * (1 + features['I1'])
            macro_interactions.append(rate_adjusted_market.fillna(0).astype(self.dtype).to_frame('rate_adjusted_market'))
        
        # 波动率-动量交互
        if all(col in features.columns for col in ['V1', 'MOM1', 'MOM2']):
            # 波动率加权的动量
            vol_weighted_momentum = features['MOM1'] / (features['V1'] + 1e-8)
            macro_interactions.append(vol_weighted_momentum.fillna(0).astype(self.dtype).to_frame('vol_weighted_momentum'))
        
        # 情绪-价格交互
        if all(col in features.columns for col in ['S1', 'P1', 'P2']):
            # 情绪调整的价格变化
            sentiment_price_interaction = features['S1'] * original_df['P1'].pct_change()
            macro_interactions.append(sentiment_price_interaction.fillna(0).astype(self.dtype).to_frame('sentiment_price_change'))
        
        # 宏观经济因子复合交互
        if all(col in features.columns for col in ['E1', 'E2', 'E3']):
            # 宏观经济强度指标
            econ_strength = (features['E1'] + features['E2'] + features['E3']) / 3
            # 宏观经济风险指标
            econ_risk = features['E1'].abs() + features['E2'].abs() + features['E3'].abs()
            macro_interactions.append(econ_strength.fillna(0).astype(self.dtype).to_frame('economic_strength'))
            macro_interactions.append(econ_risk.fillna(0).astype(self.dtype).to_frame('economic_risk'))
        
        return macro_interactions

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

    def _analyze_data_quality(self, features: pd.DataFrame) -> None:
        """分析数据质量指标"""
        
        quality_metrics = {}
        
        for col in self.numeric_columns:
            if col not in features.columns:
                continue
                
            series = features[col]
            
            # 基本统计
            quality_metrics[col] = {
                'missing_rate': series.isnull().mean(),
                'zero_rate': (series == 0).mean(),
                'unique_count': series.nunique(),
                'duplicates': series.duplicated().sum(),
                'skewness': float(series.skew()) if series.notnull().any() else 0.0,
                'kurtosis': float(series.kurtosis()) if series.notnull().any() else 0.0,
            }
            
            # 变异性指标
            if series.notnull().sum() > 1:
                cv = series.std() / abs(series.mean()) if series.mean() != 0 else np.inf
                quality_metrics[col]['coefficient_of_variation'] = cv if np.isfinite(cv) else 0.0
        
        self.data_quality_metrics = quality_metrics

    def _analyze_feature_stability(self, features: pd.DataFrame) -> None:
        """分析特征稳定性"""
        
        stability_scores = {}
        
        for col in self.numeric_columns:
            if col not in features.columns:
                continue
                
            series = features[col]
            
            if series.notnull().sum() < 10:
                stability_scores[col] = 0.0
                continue
            
            # 滚动窗口稳定性分析
            windows = [5, 10, 20]
            stability_measures = []
            
            for window in windows:
                if len(series) >= window:
                    rolling_corr = []
                    for i in range(window, len(series)):
                        slice1 = series.iloc[i-window:i]
                        slice2 = series.iloc[i-window//2:i]
                        if len(slice1) == len(slice2) and slice1.notnull().all() and slice2.notnull().all():
                            corr = slice1.corr(slice2)
                            if not pd.isna(corr):
                                rolling_corr.append(abs(corr))
                    
                    if rolling_corr:
                        stability_measures.append(np.mean(rolling_corr))
            
            # 总体稳定性得分
            if stability_measures:
                stability_scores[col] = np.mean(stability_measures)
            else:
                stability_scores[col] = 0.0
        
        self.feature_stability_scores = stability_scores

    def _perform_intelligent_feature_selection(self, features: pd.DataFrame) -> None:
        """执行智能特征选择，结合多种方法"""
        
        if not self.numeric_columns:
            return
        
        selection_scores = {}
        available_features = [col for col in self.numeric_columns if col in features.columns]
        
        if self.feature_selection_method in ["correlation", "mixed"]:
            # 基于相关性的特征选择
            correlation_scores = self._select_by_correlation(features[available_features])
            selection_scores.update(correlation_scores)
        
        if self.feature_selection_method in ["mutual_info", "mixed"]:
            # 基于互信息的特征选择
            if hasattr(self, 'target_column') and self.target_column in features.columns:
                mi_scores = self._select_by_mutual_info(features[available_features], self.target_column)
                selection_scores.update(mi_scores)
        
        if self.feature_selection_method in ["rfe", "mixed"]:
            # 基于递归特征消除的选择
            if hasattr(self, 'target_column') and self.target_column in features.columns:
                rfe_scores = self._select_by_rfe(features[available_features], self.target_column)
                selection_scores.update(rfe_scores)
        
        # 聚类分析找出冗余特征
        if self.feature_selection_method in ["mixed"]:
            cluster_analysis = self._analyze_feature_clusters(features[available_features])
            selection_scores.update(cluster_analysis)
        
        # 综合评分和选择
        self._integrate_selection_scores(selection_scores, available_features)
    
    def _select_by_correlation(self, features: pd.DataFrame) -> Dict[str, float]:
        """基于相关性分析选择特征"""
        
        if len(features.columns) < 2:
            return {col: 1.0 for col in features.columns}
        
        correlation_matrix = features.corr().abs()
        n_features = len(features.columns)
        
        # 计算每个特征的平均相关性（排除自身）
        avg_correlations = {}
        for col in features.columns:
            others = [c for c in features.columns if c != col]
            if others:
                avg_corr = correlation_matrix.loc[col, others].mean()
                avg_correlations[col] = 1.0 - avg_corr  # 相关性越低，独立性越高
            else:
                avg_correlations[col] = 1.0
        
        return avg_correlations
    
    def _select_by_mutual_info(self, features: pd.DataFrame, target: str) -> Dict[str, float]:
        """基于互信息选择特征"""
        
        if target not in features.columns:
            return {col: 0.5 for col in features.columns}
        
        target_values = features[target].values
        feature_cols = [col for col in features.columns if col != target]
        
        mi_scores = {}
        try:
            for col in feature_cols:
                if features[col].nunique() > 1:
                    # 离散化连续特征用于互信息计算
                    col_values = pd.qcut(features[col], q=5, duplicates='drop', labels=False)
                    score = mutual_info_score(target_values, col_values)
                    mi_scores[col] = score
                else:
                    mi_scores[col] = 0.0
        except Exception:
            # 如果互信息计算失败，使用默认分数
            mi_scores = {col: 0.5 for col in feature_cols}
        
        return mi_scores
    
    def _select_by_rfe(self, features: pd.DataFrame, target: str) -> Dict[str, float]:
        """基于递归特征消除选择特征"""
        
        if target not in features.columns or features[target].nunique() < 2:
            return {col: 0.5 for col in features.columns}
        
        feature_cols = [col for col in features.columns if col != target]
        if not feature_cols:
            return {}
        
        X = features[feature_cols].fillna(0)
        y = features[target]
        
        try:
            # 使用随机森林作为基估计器
            estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            
            # RFE选择一半特征
            n_features = max(1, len(feature_cols) // 2)
            rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
            rfe.fit(X, y)
            
            # 返回排名得分
            scores = {}
            for i, col in enumerate(feature_cols):
                scores[col] = 1.0 if rfe.support_[i] else 0.5
            
            return scores
        except Exception:
            # 如果RFE失败，返回默认分数
            return {col: 0.5 for col in feature_cols}
    
    def _analyze_feature_clusters(self, features: pd.DataFrame) -> Dict[str, float]:
        """分析特征聚类，找出冗余特征"""
        
        if len(features.columns) < 4:
            return {col: 0.8 for col in features.columns}
        
        # 标准化特征用于聚类
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features.fillna(0))
        
        # K-means聚类
        n_clusters = min(8, len(features.columns) // 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        # 计算每个聚类的中心特征
        cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=features.columns)
        cluster_scores = {}
        
        for i, col in enumerate(features.columns):
            cluster_id = cluster_labels[i]
            center_value = cluster_centers.iloc[cluster_id, i]
            
            # 计算与聚类中心的距离
            distance = abs(scaled_features[i] - center_value)
            avg_distance = np.mean(distance)
            
            # 距离中心越近，冗余性越高
            redundancy_score = 1.0 - avg_distance
            cluster_scores[col] = max(0.1, redundancy_score)
        
        return cluster_scores
    
    def _integrate_selection_scores(self, selection_scores: Dict[str, float], available_features: List[str]) -> None:
        """整合多种选择方法的分数"""
        
        if not selection_scores:
            self.selected_features = available_features[:self.max_features]
            return
        
        # 标准化分数
        if selection_scores:
            scores = np.array(list(selection_scores.values()))
            if len(scores) > 1:
                scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            selection_scores = dict(zip(selection_scores.keys(), scores))
        
        # 综合评分
        feature_scores = {}
        for feature in available_features:
            if feature in selection_scores:
                # 如果特征稳定性较高，给予额外加分
                stability_bonus = self.feature_stability_scores.get(feature, 0.5) * 0.2 if self.feature_stability_scores else 0
                feature_scores[feature] = selection_scores[feature] + stability_bonus
            else:
                feature_scores[feature] = 0.5
        
        # 选择最佳特征
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        selected_count = min(self.max_features, len(sorted_features))
        self.selected_features = [feat for feat, _ in sorted_features[:selected_count]]
        
        # 保存元数据
        self.selected_features_meta = {
            'total_available': len(available_features),
            'selected_count': selected_count,
            'selection_method': self.feature_selection_method,
            'feature_scores': feature_scores,
            'selection_rationale': 'Intelligent multi-method feature selection'
        }
    
    def _analyze_feature_correlations(self, features: pd.DataFrame) -> None:
        """分析特征相关性矩阵"""
        
        numeric_features = [col for col in self.numeric_columns if col in features.columns]
        
        if len(numeric_features) < 2:
            return
        
        self.correlation_matrix = features[numeric_features].corr()
        
        # 识别高相关特征对
        high_corr_pairs = []
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i + 1, len(self.correlation_matrix.columns)):
                corr_value = abs(self.correlation_matrix.iloc[i, j])
                if corr_value > 0.8:  # 高相关性阈值
                    col1 = self.correlation_matrix.columns[i]
                    col2 = self.correlation_matrix.columns[j]
                    high_corr_pairs.append((col1, col2, corr_value))
        
        # 分析特征兼容性
        self.feature_compatibility = {
            'high_correlation_pairs': high_corr_pairs,
            'correlation_threshold': 0.8,
            'feature_diversity_score': self._calculate_feature_diversity()
        }
    
    def _calculate_feature_diversity(self) -> float:
        """计算特征多样性得分"""
        
        if self.correlation_matrix is None or len(self.correlation_matrix) < 2:
            return 1.0
        
        # 计算平均绝对相关性
        upper_triangle = np.triu(self.correlation_matrix.values, k=1)
        upper_triangle = upper_triangle[upper_triangle != 0]
        avg_correlation = np.mean(np.abs(upper_triangle))
        
        # 多样性得分 = 1 - 平均相关性
        diversity_score = 1.0 - avg_correlation
        return max(0.0, min(1.0, diversity_score))
    
    def _add_intelligent_combinations(self, features: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
        """添加智能特征组合"""
        
        combination_frames = []
        
        if not self.selected_features:
            return features
        
        # 根据复杂度级别选择组合策略
        if self.combination_complexity >= 1:
            combination_frames.extend(self._add_basic_combinations(features))
        
        if self.combination_complexity >= 2:
            combination_frames.extend(self._add_polynomial_combinations(features))
        
        if self.combination_complexity >= 3:
            combination_frames.extend(self._add_conditional_combinations(features, original_df))
        
        if self.combination_complexity >= 4:
            combination_frames.extend(self._add_time_series_combinations(features))
        
        if self.combination_complexity >= 5:
            combination_frames.extend(self._add_nonlinear_combinations(features))
        
        if combination_frames:
            combination_df = pd.concat(combination_frames, axis=1)
            features = pd.concat([features, combination_df], axis=1)
        
        return features
    
    def _add_basic_combinations(self, features: pd.DataFrame) -> List[pd.DataFrame]:
        """基础特征组合"""
        
        basic_combinations = []
        selected_cols = self.selected_features or list(features.columns)[:20]
        
        # 两两组合（乘法和除法）
        for i, col1 in enumerate(selected_cols[:15]):
            for col2 in selected_cols[i+1:i+3]:  # 限制组合数量
                if col1 in features.columns and col2 in features.columns:
                    # 乘法组合
                    product = features[col1] * features[col2]
                    product = product.fillna(0).astype(self.dtype)
                    basic_combinations.append(product.to_frame(f"{col1}_x_{col2}"))
                    
                    # 除法组合（避免除零）
                    safe_divisor = features[col2].abs() + 1e-8
                    division = features[col1] / safe_divisor
                    division = division.replace([np.inf, -np.inf], 0).fillna(0)
                    basic_combinations.append(division.astype(self.dtype).to_frame(f"{col1}_div_{col2}"))
        
        return basic_combinations
    
    def _add_polynomial_combinations(self, features: pd.DataFrame) -> List[pd.DataFrame]:
        """多项式特征组合"""
        
        poly_combinations = []
        selected_cols = self.selected_features or list(features.columns)[:10]
        
        for col in selected_cols[:8]:
            if col in features.columns:
                series = features[col]
                
                # 平方
                squared = series ** 2
                poly_combinations.append(squared.fillna(0).astype(self.dtype).to_frame(f"{col}_squared"))
                
                # 立方
                cubed = series ** 3
                poly_combinations.append(cubed.fillna(0).astype(self.dtype).to_frame(f"{col}_cubed"))
                
                # 平方根
                sqrt = np.sqrt(np.abs(series))
                poly_combinations.append(sqrt.fillna(0).astype(self.dtype).to_frame(f"{col}_sqrt"))
        
        return poly_combinations
    
    def _add_conditional_combinations(self, features: pd.DataFrame, original_df: pd.DataFrame) -> List[pd.DataFrame]:
        """条件特征组合"""
        
        conditional_frames = []
        
        # 基于市场状态的条件组合
        if 'V1' in features.columns and 'P1' in original_df.columns:
            vol_state = features['V1'] > features['V1'].rolling(20).quantile(0.75)
            price_change = original_df['P1'].pct_change()
            
            # 高波动率环境下的价格动量
            high_vol_momentum = price_change.where(vol_state, 0)
            conditional_frames.append(
                high_vol_momentum.fillna(0).astype(self.dtype).to_frame('high_vol_momentum')
            )
        
        # 基于技术指标的条件组合
        if all(col in features.columns for col in ['M1', 'M2', 'M3']):
            # 市场强度条件下的特征组合
            market_strength = (features['M1'] + features['M2'] + features['M3']) / 3
            strong_market = market_strength > market_strength.rolling(20).quantile(0.8)
            
            for col in ['M1', 'M2', 'M3']:
                conditional_combo = features[col].where(strong_market, features[col] * 0.5)
                conditional_frames.append(
                    conditional_combo.fillna(0).astype(self.dtype).to_frame(f"conditional_{col}")
                )
        
        return conditional_frames
    
    def _add_time_series_combinations(self, features: pd.DataFrame) -> List[pd.DataFrame]:
        """时间序列特征组合"""
        
        ts_combinations = []
        selected_cols = self.selected_features or list(features.columns)[:8]
        
        for col in selected_cols[:6]:
            if col in features.columns and len(features) > 10:
                series = features[col]
                
                # 移动平均组合
                ma_5 = series.rolling(5).mean()
                ma_20 = series.rolling(20).mean()
                
                if self.combination_complexity >= 4:
                    ma_ratio = ma_5 / (ma_20 + 1e-8)
                    ts_combinations.append(ma_ratio.fillna(1).astype(self.dtype).to_frame(f"{col}_ma_ratio"))
                
                # 指数加权移动平均
                ewm_short = series.ewm(span=5).mean()
                ewm_long = series.ewm(span=20).mean()
                ewm_combo = ewm_short * ewm_long
                ts_combinations.append(ewm_combo.fillna(0).astype(self.dtype).to_frame(f"{col}_ewm_combo"))
        
        return ts_combinations
    
    def _add_nonlinear_combinations(self, features: pd.DataFrame) -> List[pd.DataFrame]:
        """非线性特征组合"""
        
        nonlinear_frames = []
        selected_cols = self.selected_features or list(features.columns)[:6]
        
        for col in selected_cols[:4]:
            if col in features.columns:
                series = features[col]
                
                # 对数变换
                log_series = np.log(np.abs(series) + 1e-8)
                nonlinear_frames.append(log_series.fillna(0).astype(self.dtype).to_frame(f"{col}_log"))
                
                # 指数变换
                exp_series = np.exp(series / 10)  # 缩放避免数值溢出
                nonlinear_frames.append(exp_series.fillna(1).astype(self.dtype).to_frame(f"{col}_exp"))
        
        return nonlinear_frames
    
    def _add_tiered_market_features(self, features: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
        """添加分层市场特征"""
        
        if 'V1' not in features.columns:
            return features
        
        tiered_frames = []
        
        # 市场状态分层
        market_tiers = self._define_market_tiers(features, original_df)
        
        # 为每个市场状态层级创建特征
        for tier_name, tier_condition in market_tiers.items():
            if tier_condition.sum() < 5:  # 确保有足够样本
                continue
            
            tiered_features = features.where(tier_condition)
            numeric_cols = features.select_dtypes(include=[np.number]).columns[:20]
            
            for col in numeric_cols:
                if col in features.columns:
                    # 分层均值
                    tier_mean = tiered_features[col].rolling(5, min_periods=1).mean()
                    tiered_frames.append(
                        tier_mean.fillna(features[col]).astype(self.dtype).to_frame(f"{col}_mean_{tier_name}")
                    )
                    
                    # 分层标准差
                    tier_std = tiered_features[col].rolling(5, min_periods=1).std()
                    tiered_frames.append(
                        tier_std.fillna(features[col].std()).astype(self.dtype).to_frame(f"{col}_std_{tier_name}")
                    )
        
        if tiered_frames:
            tiered_df = pd.concat(tiered_frames, axis=1)
            features = pd.concat([features, tiered_df], axis=1)
        
        return features
    
    def _define_market_tiers(self, features: pd.DataFrame, original_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """定义市场状态层级"""
        
        market_tiers = {}
        
        # 波动率层级
        if 'V1' in features.columns:
            vol_rolling = features['V1'].rolling(50, min_periods=20)
            vol_percentile = vol_rolling.rank(pct=True)
            
            market_tiers['low_vol'] = vol_percentile < 0.25
            market_tiers['normal_vol'] = (vol_percentile >= 0.25) & (vol_percentile <= 0.75)
            market_tiers['high_vol'] = vol_percentile > 0.75
        
        # 趋势层级
        if 'P1' in original_df.columns:
            price_change = original_df['P1'].pct_change(5)
            trend_strength = np.abs(price_change).rolling(20).mean()
            trend_percentile = trend_strength.rolling(50, min_periods=20).rank(pct=True)
            
            market_tiers['weak_trend'] = trend_percentile < 0.25
            market_tiers['moderate_trend'] = (trend_percentile >= 0.25) & (trend_percentile <= 0.75)
            market_tiers['strong_trend'] = trend_percentile > 0.75
        
        # 综合市场状态
        if 'low_vol' in market_tiers and 'strong_trend' in market_tiers:
            market_tiers['bull_market'] = market_tiers['low_vol'] & market_tiers['strong_trend']
            market_tiers['bear_market'] = market_tiers['high_vol'] & market_tiers['weak_trend']
        
        return market_tiers

    def _set_outlier_bounds(self, features: pd.DataFrame) -> None:
        """设置异常值边界"""
        
        for col in self.numeric_columns:
            if col not in features.columns:
                continue
                
            series = features[col].dropna()
            
            if len(series) < 10:
                continue
            
            # 使用IQR方法检测异常值
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            # 扩展IQR边界
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 使用Z-score作为补充
            z_scores = np.abs(stats.zscore(series))
            z_outliers = series[z_scores > 3]
            
            if len(z_outliers) > 0:
                # 结合IQR和Z-score边界
                z_lower = series.quantile(0.01)
                z_upper = series.quantile(0.99)
                lower_bound = min(lower_bound, z_lower)
                upper_bound = max(upper_bound, z_upper)
            
            # 确保边界合理
            if np.isfinite(lower_bound) and np.isfinite(upper_bound) and lower_bound < upper_bound:
                self.outlier_bounds[col] = (float(lower_bound), float(upper_bound))

    def _initialize_scaler(self, features: pd.DataFrame) -> None:
        """初始化智能缩放器"""
        
        try:
            # 尝试使用QuantileTransformer进行分位数标准化
            if len(self.numeric_columns) > 5:
                # 选择前20个数值特征进行缩放
                selected_cols = [col for col in self.numeric_columns if col in features.columns][:20]
                if selected_cols:
                    scaler_data = features[selected_cols].fillna(0)
                    if scaler_data.nunique().sum() > len(selected_cols):  # 确保有足够变化
                        self.scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
                        self.scaler.fit(scaler_data)
                        return
        except Exception:
            pass
        
        try:
            # 回退到RobustScaler
            if len(self.numeric_columns) > 2:
                selected_cols = [col for col in self.numeric_columns if col in features.columns][:15]
                if selected_cols:
                    scaler_data = features[selected_cols].fillna(0)
                    if scaler_data.std().sum() > 0:  # 确保有变化
                        self.scaler = RobustScaler()
                        self.scaler.fit(scaler_data)
                        return
        except Exception:
            pass
        
        # 如果都失败，保持StandardScaler
        self.scaler = None

    def _smart_fill_missing(self, series: pd.Series, col_name: str) -> pd.Series:
        """智能缺失值填充"""
        
        if series.isnull().sum() == 0:
            return series
        
        strategy = self.missing_value_strategy
        
        if strategy == "median":
            fill_value = self.fill_values.get(col_name, series.median())
            return series.fillna(fill_value)
        elif strategy == "mean":
            fill_value = series.mean()
            return series.fillna(fill_value if np.isfinite(fill_value) else 0.0)
        elif strategy == "mode":
            fill_value = series.mode()
            return series.fillna(fill_value.iloc[0] if len(fill_value) > 0 else 0.0)
        elif strategy == "ffill":
            return series.fillna(method='ffill')
        elif strategy == "bfill":
            return series.fillna(method='bfill')
        else:
            # 智能填充：结合中位数和前向填充
            median_fill = self.fill_values.get(col_name, series.median())
            series = series.fillna(median_fill)
            # 对于仍为NaN的值，使用前向填充
            return series.fillna(method='ffill').fillna(0.0)

    def get_data_quality_report(self) -> Dict[str, Any]:
        """获取数据质量报告"""
        return {
            'quality_metrics': self.data_quality_metrics,
            'stability_scores': self.feature_stability_scores,
            'outlier_bounds': self.outlier_bounds,
            'config': self.to_config()
        }

    def get_stable_features(self, threshold: float = 0.3) -> List[str]:
        """获取稳定特征列表"""
        if not self.feature_stability_scores:
            return []
        
        stable_features = [
            col for col, score in self.feature_stability_scores.items()
            if score >= threshold
        ]
        return stable_features

    def get_risky_features(self, threshold: float = 0.1) -> List[str]:
        """获取不稳定特征列表"""
        if not self.feature_stability_scores:
            return []
        
        risky_features = [
            col for col, score in self.feature_stability_scores.items()
            if score < threshold
        ]
        return risky_features

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
        """Return the canonical default pipeline configuration with intelligent optimization."""

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
            "enable_data_quality": True,
            "enable_feature_stability": True,
            "outlier_detection": True,
            "missing_value_strategy": "median",
            "enable_intelligent_selection": True,
            "enable_feature_combinations": True,
            "enable_tiered_features": True,
            "enable_robust_scaling": True,
            "feature_selection_method": "mixed",
            "combination_complexity": 3,
            "tiered_levels": 4,
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
