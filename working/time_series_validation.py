"""
时间序列友好交叉验证系统
为Hull Tactical市场预测项目提供可靠的模型验证框架

核心特性:
- 时间序列友好分割 (避免信息泄露)
- 金融特定验证策略
- 多维度验证分层
- 高级验证技术
- 智能配置和监控
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Protocol
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import warnings
import logging
import threading
import time
from datetime import datetime, timedelta
import json
from pathlib import Path
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from abc import ABC, abstractmethod

# 现有系统集成
from sklearn.model_selection import TimeSeriesSplit, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin

from .lib.models import HullModel, DynamicWeightedEnsemble
from .lib.features import FeaturePipeline
from .lib.evaluation import calculate_sharpe_ratio


class ValidationStrategy(Enum):
    """验证策略枚举"""
    TIME_SERIES_SPLIT = "time_series_split"
    EXPANDING_WINDOW = "expanding_window"
    ROLLING_WINDOW = "rolling_window"
    PURGED_TIME_SERIES = "purged_time_series"
    YEAR_BASED = "year_based"
    MARKET_REGIME_BASED = "market_regime_based"
    VOLATILITY_TIERED = "volatility_tiered"
    NESTED_CV = "nested_cv"
    PURGED_K_FOLD = "purged_k_fold"


class MarketRegime(Enum):
    """市场状态枚举"""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"


@dataclass
class ValidationConfig:
    """时间序列交叉验证配置"""
    strategy: ValidationStrategy = ValidationStrategy.TIME_SERIES_SPLIT
    n_splits: int = 5
    test_size: int = 252  # 约1年
    train_size: Optional[int] = None  # 如果为None，使用剩余数据
    gap: int = 5  # 训练和验证之间的间隔，防止信息泄露
    purge_period: int = 10  # 清理期（purged strategy）
    max_train_size: Optional[int] = None  # 扩展窗口时的最大训练大小
    
    # 金融特定参数
    min_train_samples: int = 500  # 最小训练样本数
    max_volatility_threshold: float = 3.0  # 最大波动率倍数阈值
    enable_market_regime_detection: bool = True
    enable_volatility_tiering: bool = True
    enable_drift_detection: bool = True
    
    # 高级验证参数
    enable_nested_cv: bool = False
    inner_cv_folds: int = 3
    outer_cv_folds: int = 5
    enable_purged_cv: bool = True
    embargo_percentage: float = 0.1  # 隔离期百分比
    
    # 性能监控参数
    enable_performance_monitoring: bool = True
    performance_window: int = 50
    stability_threshold: float = 0.05
    enable_early_stopping: bool = True
    early_stopping_patience: int = 3
    
    # 并行处理参数
    n_jobs: int = 1
    enable_parallel: bool = False
    random_state: int = 42
    
    # 输出控制
    verbose: bool = True
    save_splits: bool = True
    save_metrics: bool = True
    output_dir: Optional[Path] = None


@dataclass
class ValidationResult:
    """验证结果数据结构"""
    strategy: ValidationStrategy
    n_splits: int
    train_indices: List[np.ndarray] = field(default_factory=list)
    test_indices: List[np.ndarray] = field(default_factory=list)
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    feature_importance: Dict[str, List[float]] = field(default_factory=dict)
    model_predictions: Dict[int, np.ndarray] = field(default_factory=dict)
    timing_info: Dict[str, float] = field(default_factory=dict)
    stability_scores: Dict[str, float] = field(default_factory=dict)
    regime_distribution: Dict[str, int] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def add_split_result(self, split_idx: int, metrics: Dict[str, float], 
                        predictions: np.ndarray, timing: Dict[str, float]):
        """添加单个分割的结果"""
        for metric_name, value in metrics.items():
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
            self.metrics[metric_name].append(value)
        
        self.model_predictions[split_idx] = predictions
        self.timing_info.update(timing)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取验证结果摘要"""
        summary = {
            'strategy': self.strategy.value,
            'n_splits': self.n_splits,
            'timestamp': self.timestamp
        }
        
        # 计算统计指标
        for metric_name, values in self.metrics.items():
            summary[f'{metric_name}_mean'] = np.mean(values)
            summary[f'{metric_name}_std'] = np.std(values)
            summary[f'{metric_name}_min'] = np.min(values)
            summary[f'{metric_name}_max'] = np.max(values)
        
        # 添加质量指标
        summary.update(self.quality_metrics)
        
        # 添加时间信息
        if self.timing_info:
            summary['total_time'] = self.timing_info.get('total_time', 0)
            summary['avg_split_time'] = self.timing_info.get('avg_split_time', 0)
        
        return summary


class MarketRegimeDetector:
    """市场状态检测器"""
    
    def __init__(self, volatility_window: int = 20, trend_window: int = 10):
        self.volatility_window = volatility_window
        self.trend_window = trend_window
        self.regime_history = deque(maxlen=100)
        self.regime_transitions = defaultdict(int)
        
    def detect_regime(self, prices: np.ndarray, returns: np.ndarray, 
                     volatility: np.ndarray) -> MarketRegime:
        """检测当前市场状态"""
        if len(prices) < max(self.volatility_window, self.trend_window):
            return MarketRegime.UNKNOWN
        
        # 计算趋势强度
        recent_trend = np.mean(returns[-self.trend_window:])
        trend_consistency = np.mean(np.sign(returns[-self.trend_window:]) == np.sign(recent_trend))
        
        # 计算波动率状态
        recent_vol = np.std(returns[-self.volatility_window:])
        historical_vol = np.std(returns) if len(returns) > self.volatility_window else recent_vol
        vol_ratio = recent_vol / (historical_vol + 1e-8)
        
        # 动态阈值
        vol_threshold_75 = np.percentile(volatility[-50:], 75) if len(volatility) > 50 else vol_ratio
        vol_threshold_25 = np.percentile(volatility[-50:], 25) if len(volatility) > 50 else vol_ratio
        
        # 状态分类逻辑
        regime = self._classify_regime(recent_trend, trend_consistency, vol_ratio, 
                                     vol_threshold_75, vol_threshold_25)
        
        # 记录状态转移
        if self.regime_history:
            prev_regime = self.regime_history[-1]
            if prev_regime != regime:
                self.regime_transitions[f"{prev_regime.value}_to_{regime.value}"] += 1
        
        self.regime_history.append(regime)
        return regime
    
    def _classify_regime(self, trend: float, trend_consistency: float, 
                        vol_ratio: float, vol_75: float, vol_25: float) -> MarketRegime:
        """根据指标分类市场状态"""
        
        # 高波动率优先判断
        if vol_ratio > 1.5 or vol_75 > 0.02:
            if trend > 0:
                return MarketRegime.HIGH_VOLATILITY
            else:
                return MarketRegime.BEAR_MARKET
        
        # 基于趋势的状态判断
        elif trend_consistency > 0.7:
            if trend > vol_75 * 2:
                return MarketRegime.BULL_MARKET
            elif trend < -vol_75 * 2:
                return MarketRegime.BEAR_MARKET
            else:
                if vol_ratio < 0.8:
                    return MarketRegime.LOW_VOLATILITY
                else:
                    return MarketRegime.SIDEWAYS
        
        # 趋势状态
        elif trend > vol_75 * 3:
            return MarketRegime.TRENDING_UP
        elif trend < -vol_75 * 3:
            return MarketRegime.TRENDING_DOWN
        else:
            return MarketRegime.SIDEWAYS
    
    def get_regime_stability(self) -> float:
        """计算市场状态稳定性"""
        if len(self.regime_history) < 5:
            return 0.0
        
        recent_regimes = list(self.regime_history)[-10:]
        regime_counts = defaultdict(int)
        for regime in recent_regimes:
            regime_counts[regime] += 1
        
        # 状态集中度作为稳定性指标
        max_count = max(regime_counts.values())
        stability = max_count / len(recent_regimes)
        return min(1.0, stability)


class TemporalSplitter:
    """时间序列分割器 - 支持多种策略"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.regime_detector = MarketRegimeDetector()
        self.split_history = []
        
    def split(self, X: pd.DataFrame, y: pd.Series, 
             additional_data: Optional[Dict] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """执行时间序列分割"""
        
        start_time = time.time()
        n_samples = len(X)
        
        if self.config.strategy == ValidationStrategy.TIME_SERIES_SPLIT:
            splits = self._time_series_split(X, y, additional_data)
        elif self.config.strategy == ValidationStrategy.EXPANDING_WINDOW:
            splits = self._expanding_window(X, y, additional_data)
        elif self.config.strategy == ValidationStrategy.ROLLING_WINDOW:
            splits = self._rolling_window(X, y, additional_data)
        elif self.config.strategy == ValidationStrategy.PURGED_TIME_SERIES:
            splits = self._purged_time_series(X, y, additional_data)
        elif self.config.strategy == ValidationStrategy.YEAR_BASED:
            splits = self._year_based_split(X, y, additional_data)
        elif self.config.strategy == ValidationStrategy.MARKET_REGIME_BASED:
            splits = self._market_regime_based_split(X, y, additional_data)
        elif self.config.strategy == ValidationStrategy.VOLATILITY_TIERED:
            splits = self._volatility_tiered_split(X, y, additional_data)
        elif self.config.strategy == ValidationStrategy.PURGED_K_FOLD:
            splits = self._purged_k_fold(X, y, additional_data)
        else:
            raise ValueError(f"不支持的验证策略: {self.config.strategy}")
        
        # 记录分割历史
        self.split_history.append({
            'timestamp': datetime.now().isoformat(),
            'strategy': self.config.strategy.value,
            'n_splits': len(splits),
            'total_time': time.time() - start_time,
            'n_samples': n_samples
        })
        
        return splits
    
    def _time_series_split(self, X: pd.DataFrame, y: pd.Series, 
                          additional_data: Optional[Dict]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """标准时间序列分割"""
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits, 
                              test_size=self.config.test_size,
                              gap=self.config.gap)
        return list(tscv.split(X))
    
    def _expanding_window(self, X: pd.DataFrame, y: pd.Series, 
                         additional_data: Optional[Dict]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """扩展窗口分割"""
        splits = []
        n_samples = len(X)
        
        # 计算每个验证集的起始位置
        val_starts = np.linspace(
            self.config.min_train_samples,
            n_samples - self.config.test_size,
            self.config.n_splits,
            dtype=int
        )
        
        for val_start in val_starts:
            train_end = val_start - self.config.gap
            train_start = 0
            val_end = min(val_start + self.config.test_size, n_samples)
            
            # 确保最小训练大小
            if train_end - train_start < self.config.min_train_samples:
                continue
            
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(val_start, val_end)
            
            splits.append((train_idx, test_idx))
        
        return splits
    
    def _rolling_window(self, X: pd.DataFrame, y: pd.Series, 
                       additional_data: Optional[Dict]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """滚动窗口分割"""
        splits = []
        n_samples = len(X)
        
        # 固定训练大小
        if self.config.train_size is None:
            self.config.train_size = max(self.config.min_train_samples, self.config.test_size * 2)
        
        val_starts = np.linspace(
            self.config.train_size + self.config.gap,
            n_samples - self.config.test_size,
            self.config.n_splits,
            dtype=int
        )
        
        for val_start in val_starts:
            train_start = val_start - self.config.gap - self.config.train_size
            train_end = val_start - self.config.gap
            val_end = min(val_start + self.config.test_size, n_samples)
            
            # 确保窗口大小
            if train_end - train_start != self.config.train_size:
                continue
            
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(val_start, val_end)
            
            splits.append((train_idx, test_idx))
        
        return splits
    
    def _purged_time_series(self, X: pd.DataFrame, y: pd.Series, 
                           additional_data: Optional[Dict]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """清理时间序列分割，避免信息重叠"""
        splits = []
        n_samples = len(X)
        
        # 计算purge大小
        purge_size = int(self.config.test_size * self.config.embargo_percentage)
        total_purge = self.config.gap + purge_size
        
        val_starts = np.linspace(
            self.config.min_train_samples,
            n_samples - self.config.test_size - total_purge,
            self.config.n_splits,
            dtype=int
        )
        
        for val_start in val_starts:
            train_end = val_start - self.config.gap
            train_start = 0
            val_end = min(val_start + self.config.test_size, n_samples)
            purge_end = min(val_end + purge_size, n_samples)
            
            # 训练集：排除purge期间
            train_idx = np.arange(train_start, train_end)
            # 验证集：包含purge后的数据
            test_idx = np.arange(val_start, val_end)
            
            splits.append((train_idx, test_idx))
        
        return splits
    
    def _year_based_split(self, X: pd.DataFrame, y: pd.Series, 
                         additional_data: Optional[Dict]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """基于年份的分割"""
        if 'date_id' not in X.columns:
            # 如果没有date_id，回退到时间序列分割
            return self._time_series_split(X, y, additional_data)
        
        splits = []
        date_ids = X['date_id'].values
        
        # 按年份分组
        unique_years = sorted(set(date_ids // 10000))  # 假设date_id格式为YYYYMMDD
        test_years = unique_years[-self.config.n_splits:] if len(unique_years) >= self.config.n_splits else unique_years[-1:]
        
        for i, test_year in enumerate(test_years):
            # 训练集：之前所有年份
            train_mask = date_ids < (test_year * 10000)
            # 验证集：测试年份
            test_mask = (date_ids >= (test_year * 10000)) & (date_ids < ((test_year + 1) * 10000))
            
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            
            if len(train_idx) >= self.config.min_train_samples and len(test_idx) > 0:
                splits.append((train_idx, test_idx))
        
        return splits
    
    def _market_regime_based_split(self, X: pd.DataFrame, y: pd.Series, 
                                  additional_data: Optional[Dict]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """基于市场状态的分割"""
        if additional_data is None or 'prices' not in additional_data:
            return self._time_series_split(X, y, additional_data)
        
        prices = additional_data['prices']
        returns = np.diff(prices) / prices[:-1]
        volatility = additional_data.get('volatility', np.abs(returns))
        
        # 为每个样本分配市场状态
        regimes = []
        for i in range(len(prices)):
            regime = self.regime_detector.detect_regime(
                prices[:i+1], 
                returns[:i] if i > 0 else np.array([]),
                volatility[:i] if i > 0 else np.array([])
            )
            regimes.append(regime)
        
        # 按市场状态分组
        regime_groups = defaultdict(list)
        for i, regime in enumerate(regimes):
            regime_groups[regime].append(i)
        
        # 创建基于状态的训练-验证分割
        splits = []
        regime_list = list(regime_groups.keys())
        
        for i in range(min(self.config.n_splits, len(regime_list))):
            test_regime = regime_list[i]
            train_regimes = regime_list[:i] + regime_list[i+1:]
            
            train_idx = np.concatenate([regime_groups[r] for r in train_regimes])
            test_idx = np.array(regime_groups[test_regime])
            
            if len(train_idx) >= self.config.min_train_samples and len(test_idx) > 0:
                splits.append((train_idx, test_idx))
        
        return splits if splits else self._time_series_split(X, y, additional_data)
    
    def _volatility_tiered_split(self, X: pd.DataFrame, y: pd.Series, 
                                additional_data: Optional[Dict]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """基于波动率分层的分割"""
        if additional_data is None or 'volatility' not in additional_data:
            return self._time_series_split(X, y, additional_data)
        
        volatility = additional_data['volatility']
        
        # 计算波动率分位数
        vol_quantiles = np.quantile(volatility, [0.33, 0.67])
        
        # 为每个样本分配波动率层级
        low_vol_mask = volatility <= vol_quantiles[0]
        med_vol_mask = (volatility > vol_quantiles[0]) & (volatility <= vol_quantiles[1])
        high_vol_mask = volatility > vol_quantiles[1]
        
        splits = []
        
        # 创建分层验证集
        all_indices = np.arange(len(volatility))
        vol_tiers = [low_vol_mask, med_vol_mask, high_vol_mask]
        tier_names = ['low_vol', 'med_vol', 'high_vol']
        
        for i in range(min(self.config.n_splits, len(vol_tiers))):
            test_indices = all_indices[vol_tiers[i]]
            
            # 训练集：其他所有层级
            train_indices = all_indices[~vol_tiers[i]]
            
            if len(train_indices) >= self.config.min_train_samples and len(test_indices) > 0:
                splits.append((train_indices, test_indices))
        
        return splits if splits else self._time_series_split(X, y, additional_data)
    
    def _purged_k_fold(self, X: pd.DataFrame, y: pd.Series, 
                      additional_data: Optional[Dict]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """清理K折交叉验证"""
        n_samples = len(X)
        fold_size = n_samples // self.config.n_splits
        purge_size = int(fold_size * self.config.embargo_percentage)
        
        indices = np.arange(n_samples)
        splits = []
        
        for i in range(self.config.n_splits):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < self.config.n_splits - 1 else n_samples
            
            # 验证集
            val_start = start_idx
            val_end = end_idx
            
            # 训练集：排除验证集和purge期间
            train_start = 0
            train_end = start_idx - self.config.gap - purge_size
            train_start_purge = end_idx + purge_size
            train_end_purge = n_samples
            
            train_indices = []
            if train_end > train_start:
                train_indices.extend(indices[train_start:train_end])
            if train_end_purge > train_start_purge:
                train_indices.extend(indices[train_start_purge:train_end_purge])
            
            val_indices = indices[val_start:val_end]
            
            if len(train_indices) >= self.config.min_train_samples and len(val_indices) > 0:
                splits.append((np.array(train_indices), val_indices))
        
        return splits


class FinanceValidationMetrics:
    """金融专用验证指标计算器"""
    
    def __init__(self, risk_free_rate: float = 0.02, target_volatility: float = 0.20):
        self.risk_free_rate = risk_free_rate
        self.target_volatility = target_volatility
        self.max_leverage = 2.0  # 最大杠杆倍数
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         benchmark_returns: Optional[np.ndarray] = None,
                         market_returns: Optional[np.ndarray] = None) -> Dict[str, float]:
        """计算综合金融指标"""
        
        metrics = {}
        
        # 基础回归指标
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # 金融特定指标
        if len(y_pred) > 1:
            # 计算策略收益
            strategy_returns = self._calculate_strategy_returns(y_pred)
            metrics['strategy_sharpe'] = self._calculate_sharpe(strategy_returns)
            metrics['strategy_volatility'] = np.std(strategy_returns)
            metrics['strategy_max_drawdown'] = self._calculate_max_drawdown(strategy_returns)
            
            # 与基准对比
            if benchmark_returns is not None:
                metrics['excess_return'] = np.mean(strategy_returns) - np.mean(benchmark_returns)
                metrics['information_ratio'] = self._calculate_information_ratio(strategy_returns, benchmark_returns)
                metrics['tracking_error'] = np.std(strategy_returns - benchmark_returns)
            
            # 市场相关性
            if market_returns is not None:
                metrics['market_correlation'] = np.corrcoef(strategy_returns, market_returns)[0, 1]
            
            # 风险调整指标
            metrics['calmar_ratio'] = metrics['strategy_sharpe'] / (metrics['strategy_max_drawdown'] + 1e-8)
            metrics['sortino_ratio'] = self._calculate_sortino(strategy_returns)
        
        # 预测质量指标
        metrics['directional_accuracy'] = self._calculate_directional_accuracy(y_true, y_pred)
        metrics['prediction_bias'] = np.mean(y_pred) - np.mean(y_true)
        metrics['prediction_std_ratio'] = np.std(y_pred) / (np.std(y_true) + 1e-8)
        
        return metrics
    
    def _calculate_strategy_returns(self, predictions: np.ndarray) -> np.ndarray:
        """计算策略收益（简化的仓位-收益关系）"""
        # 将预测转换为仓位（0-2范围）
        positions = np.clip(predictions, 0, self.max_leverage)
        
        # 假设单位收益（实际中应使用真实市场收益）
        if len(positions) > 1:
            unit_returns = np.diff(positions) / (positions[:-1] + 1e-8)
        else:
            unit_returns = np.array([0.0])
        
        return unit_returns
    
    def _calculate_sharpe(self, returns: np.ndarray, period: str = 'daily') -> float:
        """计算夏普比率"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / 252  # 假设252个交易日
        return np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)
    
    def _calculate_sortino(self, returns: np.ndarray) -> float:
        """计算索提诺比率（只考虑负波动率）"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / 252
        negative_returns = excess_returns[excess_returns < 0]
        
        if len(negative_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
        
        downside_std = np.std(negative_returns)
        return np.mean(excess_returns) / (downside_std + 1e-8)
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """计算最大回撤"""
        if len(returns) == 0:
            return 0.0
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return abs(np.min(drawdown))
    
    def _calculate_information_ratio(self, strategy_returns: np.ndarray, 
                                   benchmark_returns: np.ndarray) -> float:
        """计算信息比率"""
        if len(strategy_returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        # 确保长度一致
        min_len = min(len(strategy_returns), len(benchmark_returns))
        strategy_returns = strategy_returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]
        
        excess_returns = strategy_returns - benchmark_returns
        return np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)
    
    def _calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算方向预测准确率"""
        if len(y_true) < 2 or len(y_pred) < 2:
            return 0.0
        
        # 计算变化方向
        true_changes = np.diff(y_true) > 0
        pred_changes = np.diff(y_pred) > 0
        
        accuracy = np.mean(true_changes == pred_changes)
        return accuracy


class PerformanceMonitor:
    """验证性能监控器"""
    
    def __init__(self, window_size: int = 50, stability_threshold: float = 0.05):
        self.window_size = window_size
        self.stability_threshold = stability_threshold
        self.metric_history = defaultdict(deque)
        self.stability_scores = defaultdict(float)
        self.alert_conditions = []
        
    def update_metrics(self, split_idx: int, metrics: Dict[str, float]):
        """更新性能指标"""
        for metric_name, value in metrics.items():
            if metric_name in ['mse', 'mae']:  # 越小越好的指标
                self.metric_history[metric_name].append(value)
            else:  # 越大越好的指标
                self.metric_history[metric_name].append(-value)  # 取负值以便于稳定性计算
        
        # 保持窗口大小
        for metric_name in self.metric_history:
            if len(self.metric_history[metric_name]) > self.window_size:
                self.metric_history[metric_name].popleft()
        
        # 更新稳定性分数
        self._update_stability_scores()
        
        # 检查告警条件
        self._check_alert_conditions()
    
    def _update_stability_scores(self):
        """更新稳定性分数"""
        for metric_name, history in self.metric_history.items():
            if len(history) < 5:  # 需要至少5个数据点
                self.stability_scores[metric_name] = 0.0
                continue
            
            # 计算最近趋势的方差
            recent_values = list(history)[-10:]  # 最近10个值
            if len(recent_values) >= 3:
                trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                stability = 1.0 / (1.0 + abs(trend) * 10)  # 趋势越平缓，稳定性越高
                self.stability_scores[metric_name] = min(1.0, stability)
    
    def _check_alert_conditions(self):
        """检查告警条件"""
        self.alert_conditions = []
        
        for metric_name, score in self.stability_scores.items():
            if score < self.stability_threshold:
                self.alert_conditions.append(f"指标 {metric_name} 稳定性不足: {score:.3f}")
    
    def get_stability_report(self) -> Dict[str, Any]:
        """获取稳定性报告"""
        return {
            'stability_scores': dict(self.stability_scores),
            'alert_conditions': self.alert_conditions,
            'monitoring_active': len(self.metric_history) > 0,
            'window_size': self.window_size,
            'stability_threshold': self.stability_threshold
        }


class ValidationScheduler:
    """验证策略调度器"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.schedule_history = []
        self.performance_tracker = PerformanceMonitor()
        self.current_strategy = config.strategy
        
    def auto_select_strategy(self, data_characteristics: Dict[str, Any]) -> ValidationStrategy:
        """基于数据特征自动选择最佳验证策略"""
        
        n_samples = data_characteristics.get('n_samples', 0)
        has_market_data = data_characteristics.get('has_market_data', False)
        volatility_range = data_characteristics.get('volatility_range', (0, 1))
        trend_strength = data_characteristics.get('trend_strength', 0)
        
        # 策略选择逻辑
        if n_samples < 1000:
            return ValidationStrategy.TIME_SERIES_SPLIT
        elif has_market_data and volatility_range[1] - volatility_range[0] > 0.5:
            return ValidationStrategy.VOLATILITY_TIERED
        elif abs(trend_strength) > 0.3:
            return ValidationStrategy.MARKET_REGIME_BASED
        elif n_samples > 2000:
            return ValidationStrategy.EXPANDING_WINDOW
        else:
            return ValidationStrategy.PURGED_TIME_SERIES
    
    def get_recommended_config(self, data_info: Dict[str, Any]) -> ValidationConfig:
        """基于数据信息推荐配置"""
        config = ValidationConfig()
        
        n_samples = data_info.get('n_samples', 1000)
        
        # 调整样本数量
        if n_samples < 500:
            config.n_splits = 3
            config.test_size = max(50, n_samples // 20)
        elif n_samples > 5000:
            config.n_splits = 7
            config.test_size = min(500, n_samples // 10)
        else:
            config.n_splits = 5
            config.test_size = min(200, n_samples // 15)
        
        # 启用高级策略
        if n_samples > 1000:
            config.enable_nested_cv = True
            config.enable_purged_cv = True
        
        return config
    
    def schedule_validation_round(self, round_num: int, total_rounds: int) -> ValidationConfig:
        """调度验证轮次"""
        # 根据轮次调整策略
        if round_num == 0:
            # 第一轮：保守策略
            return ValidationConfig(
                strategy=ValidationStrategy.TIME_SERIES_SPLIT,
                n_splits=3,
                verbose=True
            )
        elif round_num == total_rounds - 1:
            # 最后一轮：全面验证
            return ValidationConfig(
                strategy=ValidationStrategy.EXPANDING_WINDOW,
                n_splits=7,
                enable_nested_cv=True,
                enable_purged_cv=True
            )
        else:
            # 中间轮次：平衡策略
            return ValidationConfig(
                strategy=ValidationStrategy.MARKET_REGIME_BASED,
                n_splits=5,
                enable_performance_monitoring=True
            )
    
    def get_performance_feedback(self) -> Dict[str, Any]:
        """获取性能反馈用于策略调整"""
        return self.performance_tracker.get_stability_report()


class TimeSeriesCrossValidator:
    """时间序列交叉验证主类"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.splitter = TemporalSplitter(config)
        self.metrics_calculator = FinanceValidationMetrics()
        self.scheduler = ValidationScheduler(config)
        
        # 线程安全
        self.lock = threading.RLock()
        
        # 结果存储
        self.results: List[ValidationResult] = []
        self.split_statistics = defaultdict(list)
        
        # 设置日志
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def validate(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                additional_data: Optional[Dict] = None,
                return_predictions: bool = True) -> ValidationResult:
        """执行时间序列交叉验证"""
        
        start_time = time.time()
        self.logger.info(f"开始时间序列交叉验证: {self.config.strategy.value}")
        
        # 获取分割
        splits = self.splitter.split(X, y, additional_data)
        
        if not splits:
            raise ValueError("无法生成有效的分割")
        
        # 初始化结果
        result = ValidationResult(
            strategy=self.config.strategy,
            n_splits=len(splits),
            train_indices=[train_idx for train_idx, _ in splits],
            test_indices=[test_idx for _, test_idx in splits]
        )
        
        # 存储分割统计
        for i, (train_idx, test_idx) in enumerate(splits):
            self.split_statistics['train_size'].append(len(train_idx))
            self.split_statistics['test_size'].append(len(test_idx))
            self.split_statistics['split_ratio'].append(len(test_idx) / len(train_idx))
        
        # 并行或串行验证
        if self.config.enable_parallel and self.config.n_jobs > 1:
            fold_results = self._parallel_validation(model, X, y, splits)
        else:
            fold_results = self._sequential_validation(model, X, y, splits)
        
        # 合并结果
        for split_idx, (metrics, predictions, timing) in enumerate(fold_results):
            result.add_split_result(split_idx, metrics, predictions, timing)
            
            # 性能监控
            if self.config.enable_performance_monitoring:
                self.scheduler.performance_tracker.update_metrics(split_idx, metrics)
        
        # 计算质量指标
        result.quality_metrics = self._calculate_quality_metrics(result)
        
        # 计算总体时间
        total_time = time.time() - start_time
        result.timing_info['total_time'] = total_time
        result.timing_info['avg_split_time'] = total_time / len(splits)
        
        # 稳定性分析
        if self.config.enable_drift_detection:
            result.stability_scores = self._analyze_stability(result)
        
        # 市场状态分布分析
        if self.config.enable_market_regime_detection and additional_data:
            result.regime_distribution = self._analyze_regime_distribution(splits, additional_data)
        
        self.logger.info(f"验证完成: {len(splits)} 折, 总时间: {total_time:.2f}秒")
        
        return result
    
    def _sequential_validation(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                              splits: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[Dict, np.ndarray, Dict]]:
        """串行验证"""
        results = []
        
        for split_idx, (train_idx, test_idx) in enumerate(splits):
            self.logger.info(f"执行分割 {split_idx + 1}/{len(splits)}")
            
            split_start = time.time()
            
            try:
                # 分割数据
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # 训练模型
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_train, y_train)
                
                # 预测
                predictions = model.predict(X_test)
                
                # 计算指标
                additional_info = {}
                if 'market_returns' in X.columns:
                    additional_info['market_returns'] = X_test['market_returns'].values
                if 'benchmark_returns' in X.columns:
                    additional_info['benchmark_returns'] = X_test['benchmark_returns'].values
                
                metrics = self.metrics_calculator.calculate_metrics(
                    y_test.values, predictions, **additional_info
                )
                
                timing = {
                    'split_time': time.time() - split_start,
                    'fit_time': getattr(model, 'last_fit_time', 0),
                    'predict_time': getattr(model, 'last_predict_time', 0)
                }
                
                results.append((metrics, predictions, timing))
                
            except Exception as e:
                self.logger.error(f"分割 {split_idx} 执行失败: {e}")
                # 记录失败但继续其他分割
                results.append(({'error': str(e)}, np.zeros(len(test_idx)), {}))
        
        return results
    
    def _parallel_validation(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                            splits: List[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[Dict, np.ndarray, Dict]]:
        """并行验证"""
        results = [None] * len(splits)
        
        def process_split(args):
            split_idx, train_idx, test_idx = args
            try:
                # 分割数据
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # 重新创建模型避免状态冲突
                temp_model = type(model)(**getattr(model, 'get_params', lambda: {})())
                
                # 训练预测
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    temp_model.fit(X_train, y_train)
                    predictions = temp_model.predict(X_test)
                
                # 计算指标
                additional_info = {}
                if 'market_returns' in X.columns:
                    additional_info['market_returns'] = X_test['market_returns'].values
                if 'benchmark_returns' in X.columns:
                    additional_info['benchmark_returns'] = X_test['benchmark_returns'].values
                
                metrics = self.metrics_calculator.calculate_metrics(
                    y_test.values, predictions, **additional_info
                )
                
                return split_idx, metrics, predictions
                
            except Exception as e:
                self.logger.error(f"并行分割 {split_idx} 执行失败: {e}")
                return split_idx, {'error': str(e)}, np.zeros(len(test_idx))
        
        # 并行处理
        with ThreadPoolExecutor(max_workers=self.config.n_jobs) as executor:
            futures = [
                executor.submit(process_split, (i, train_idx, test_idx))
                for i, (train_idx, test_idx) in enumerate(splits)
            ]
            
            for future in as_completed(futures):
                split_idx, metrics, predictions = future.result()
                results[split_idx] = (metrics, predictions, {})
        
        return results
    
    def _calculate_quality_metrics(self, result: ValidationResult) -> Dict[str, float]:
        """计算验证质量指标"""
        quality = {}
        
        # 分割均匀性
        train_sizes = [len(idx) for idx in result.train_indices]
        test_sizes = [len(idx) for idx in result.test_indices]
        
        quality['train_size_cv'] = np.std(train_sizes) / (np.mean(train_sizes) + 1e-8)
        quality['test_size_cv'] = np.std(test_sizes) / (np.mean(test_sizes) + 1e-8)
        
        # 性能一致性
        if result.metrics:
            primary_metric = 'mse' if 'mse' in result.metrics else list(result.metrics.keys())[0]
            scores = result.metrics[primary_metric]
            quality['performance_stability'] = 1.0 / (1.0 + np.std(scores) / (np.mean(scores) + 1e-8))
        
        # 样本利用率
        total_samples = sum(train_sizes) + sum(test_sizes)
        if len(result.train_indices) > 0:
            avg_train_size = np.mean(train_sizes)
            avg_test_size = np.mean(test_sizes)
            quality['sample_efficiency'] = avg_test_size / (avg_train_size + avg_test_size)
        
        return quality
    
    def _analyze_stability(self, result: ValidationResult) -> Dict[str, float]:
        """分析验证稳定性"""
        stability = {}
        
        for metric_name, values in result.metrics.items():
            if len(values) >= 3:
                # 计算最近趋势
                recent_values = values[-5:] if len(values) >= 5 else values
                if len(recent_values) >= 3:
                    trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                    stability[f'{metric_name}_trend'] = trend
                    stability[f'{metric_name}_stability'] = 1.0 / (1.0 + abs(trend) * 10)
        
        return stability
    
    def _analyze_regime_distribution(self, splits: List[Tuple[np.ndarray, np.ndarray]], 
                                   additional_data: Dict) -> Dict[str, int]:
        """分析市场状态分布"""
        if not additional_data or 'regimes' not in additional_data:
            return {}
        
        regimes = additional_data['regimes']
        distribution = defaultdict(int)
        
        for train_idx, test_idx in splits:
            for idx in test_idx:
                if idx < len(regimes):
                    regime = regimes[idx]
                    distribution[regime] += 1
        
        return dict(distribution)
    
    def compare_strategies(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                          strategies: List[ValidationStrategy],
                          additional_data: Optional[Dict] = None) -> Dict[ValidationStrategy, ValidationResult]:
        """比较不同验证策略的效果"""
        results = {}
        
        for strategy in strategies:
            self.logger.info(f"测试策略: {strategy.value}")
            
            # 创建策略配置
            strategy_config = ValidationConfig(
                strategy=strategy,
                n_splits=self.config.n_splits,
                test_size=self.config.test_size,
                verbose=self.config.verbose
            )
            
            # 执行验证
            validator = TimeSeriesCrossValidator(strategy_config)
            result = validator.validate(model, X, y, additional_data)
            results[strategy] = result
        
        return results
    
    def save_results(self, result: ValidationResult, output_path: Path):
        """保存验证结果"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为可序列化格式
        serializable_result = {
            'strategy': result.strategy.value,
            'n_splits': result.n_splits,
            'metrics': result.metrics,
            'quality_metrics': result.quality_metrics,
            'stability_scores': result.stability_scores,
            'regime_distribution': result.regime_distribution,
            'timing_info': result.timing_info,
            'timestamp': result.timestamp
        }
        
        # 保存JSON
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(serializable_result, f, indent=2, default=str)
        
        # 保存详细结果
        with open(output_path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(result, f)
        
        self.logger.info(f"结果已保存到: {output_path}")


# 便利函数
def create_time_series_validator(strategy: Union[str, ValidationStrategy] = 'time_series_split',
                                n_splits: int = 5, 
                                **kwargs) -> TimeSeriesCrossValidator:
    """创建时间序列验证器的便利函数"""
    if isinstance(strategy, str):
        strategy = ValidationStrategy(strategy)
    
    config = ValidationConfig(
        strategy=strategy,
        n_splits=n_splits,
        **kwargs
    )
    
    return TimeSeriesCrossValidator(config)


def validate_model_time_series(model: BaseEstimator, 
                              X: pd.DataFrame, 
                              y: pd.Series,
                              strategy: str = 'time_series_split',
                              **kwargs) -> ValidationResult:
    """便利函数：快速验证模型"""
    validator = create_time_series_validator(strategy, **kwargs)
    return validator.validate(model, X, y)


# 导出
__all__ = [
    'TimeSeriesCrossValidator',
    'ValidationConfig', 
    'ValidationResult',
    'ValidationStrategy',
    'MarketRegime',
    'FinanceValidationMetrics',
    'TemporalSplitter',
    'ValidationScheduler',
    'create_time_series_validator',
    'validate_model_time_series'
]