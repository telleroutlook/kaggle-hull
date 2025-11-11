"""
模型定义和训练工具
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Sequence, Union, Tuple, Callable
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import deque, defaultdict
from datetime import datetime, timedelta
import warnings
import threading
import inspect
from concurrent.futures import ThreadPoolExecutor


def create_baseline_model(random_state: Optional[int] = 42, **overrides):
    """创建基线模型"""

    # 导入配置
    try:
        from .config import get_config
        config = get_config()
        model_config = config.get_model_config()
        default_params = {
            "n_estimators": model_config['baseline_n_estimators'],
            "max_depth": model_config['baseline_max_depth'],
            "random_state": model_config['baseline_random_state'],
        }
    except ImportError:
        default_params = {
            "n_estimators": 200,
            "max_depth": 12,
            "random_state": 42,
        }

    if random_state is not None:
        default_params["random_state"] = random_state

    default_params.update(overrides)

    try:
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=default_params.get("n_estimators", 200),
            max_depth=default_params.get("max_depth", 12),
            random_state=default_params.get("random_state", random_state or 42),
            min_samples_leaf=default_params.get("min_samples_leaf", 5),
            n_jobs=-1,
        )
    except ImportError:
        # 如果scikit-learn不可用，返回简单模型
        class SimpleModel:
            def __init__(self):
                self.coef_ = None

            def fit(self, X, y):
                X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
                self.coef_ = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]

            def predict(self, X):
                X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
                return X_with_intercept @ self.coef_

        return SimpleModel()


def create_lightgbm_model(random_state: Optional[int] = 42, **overrides):
    """LightGBM 模型"""

    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise ImportError("LightGBM 未安装") from exc

    params = {
        "objective": "regression",
        "n_estimators": 2000,
        "learning_rate": 0.015,
        "num_leaves": 256,
        "max_depth": 10,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 0.8,
        "min_child_samples": 8,
        "n_jobs": -1,
        "verbosity": -1,
    }
    if random_state is not None:
        params.setdefault("random_state", random_state)
        params.setdefault("seed", random_state)
    params.update(overrides)
    return LGBMRegressor(**params)


def create_xgboost_model(random_state: Optional[int] = 42, **overrides):
    """XGBoost 模型"""

    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("XGBoost 未安装") from exc

    params = {
        "n_estimators": 3000,
        "learning_rate": 0.01,
        "max_depth": 7,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_lambda": 2.0,
        "reg_alpha": 0.0,
        "gamma": 0.0,
        "min_child_weight": 10,
        "tree_method": "hist",
        "n_jobs": -1,
    }
    if random_state is not None:
        params.setdefault("random_state", random_state)
        params.setdefault("seed", random_state)
    params.update(overrides)
    return XGBRegressor(**params)


def create_catboost_model(random_state: Optional[int] = 42, **overrides):
    """CatBoost 回归模型"""

    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("CatBoost 未安装") from exc

    params = {
        "depth": 7,
        "learning_rate": 0.02,
        "l2_leaf_reg": 4.0,
        "iterations": 5000,
        "loss_function": "RMSE",
        "bagging_temperature": 0.8,
        "random_strength": 0.8,
        "allow_writing_files": False,
        "thread_count": -1,
        "verbose": False,
    }
    if random_state is not None:
        params.setdefault("random_seed", random_state)
    params.update(overrides)
    return CatBoostRegressor(**params)


class AdvancedModelPerformanceMonitor:
    """增强模型性能监控器 - 支持时间窗口自适应和市场状态感知"""
    
    def __init__(self, window_size: int = 100, update_frequency: int = 10,
                 adaptation_threshold: float = 0.05, market_regime_detection: bool = True):
        self.base_window_size = window_size
        self.window_size = window_size
        self.update_frequency = update_frequency
        self.adaptation_threshold = adaptation_threshold
        self.market_regime_detection = market_regime_detection
        
        self.model_performances = defaultdict(lambda: deque(maxlen=window_size))
        self.model_stability_scores = defaultdict(float)
        self.market_regime_history = deque(maxlen=50)
        self.performance_trends = defaultdict(list)
        self.last_update_idx = 0
        self.lock = threading.RLock()
        
        # 自适应参数
        self.regime_detection_window = 20
        self.stability_lookback = 10
        self.trend_analysis_window = 30
        
    def _detect_market_regime(self, true_values: np.ndarray) -> str:
        """检测市场状态：trending_up, trending_down, volatile, stable"""
        if not self.market_regime_detection or len(true_values) < 5:
            return 'unknown'
            
        # 计算市场特征
        returns = np.diff(true_values) if len(true_values) > 1 else true_values
        volatility = np.std(returns[-min(10, len(returns)):]) if len(returns) > 0 else 0.0
        trend = np.mean(returns[-min(5, len(returns)):]) if len(returns) > 0 else 0.0
        
        # 动态阈值基于历史波动性
        # 使用固定默认值避免类型错误
        historical_vol = 0.02
        if len(self.market_regime_history) > 5:
            # 如果需要历史波动性，可以基于之前的returns计算
            historical_vol = np.std(returns[-20:]) if len(returns) >= 20 else 0.02
        
        vol_threshold = max(0.01, historical_vol * 1.5)
        
        # 状态分类
        if volatility > vol_threshold * 2:
            regime = 'volatile'
        elif trend > vol_threshold * 3:
            regime = 'trending_up' 
        elif trend < -vol_threshold * 3:
            regime = 'trending_down'
        else:
            regime = 'stable'
            
        self.market_regime_history.append(regime)
        return regime
    
    def _update_adaptive_window(self, model_name: str, current_performance: float):
        """自适应调整性能监控窗口大小"""
        if len(self.model_performances[model_name]) < 5:
            return
            
        recent_performances = [p['mse'] for p in list(self.model_performances[model_name])[-10:]]
        performance_variance = np.var(recent_performances)
        
        # 基于性能方差调整窗口大小
        if performance_variance > self.adaptation_threshold:
            # 高方差：增加窗口大小以平滑波动
            self.window_size = min(self.base_window_size * 1.5, 200)
        else:
            # 低方差：保持或减小窗口以快速响应
            self.window_size = max(self.base_window_size * 0.8, 50)
            
    def _calculate_stability_score(self, model_name: str) -> float:
        """计算模型稳定性分数"""
        if len(self.model_performances[model_name]) < 3:
            return 1.0
            
        recent_perf = list(self.model_performances[model_name])[-self.stability_lookback:]
        mse_values = [p['mse'] for p in recent_perf]
        
        # 计算性能变异性系数
        cv = np.std(mse_values) / (np.mean(mse_values) + 1e-8)
        stability = 1.0 / (1.0 + cv)
        
        # 趋势分析：性能是否在改善
        if len(mse_values) >= 5:
            recent_trend = np.polyfit(range(len(mse_values[-5:])), mse_values[-5:], 1)[0]
            trend_bonus = 0.1 if recent_trend < 0 else 0.0  # 下降趋势奖励
            stability += trend_bonus
            
        self.model_stability_scores[model_name] = stability
        return min(stability, 1.0)
    
    def update_performance(self, model_name: str, true_values: np.ndarray, 
                          predictions: np.ndarray, update_idx: int):
        """更新模型性能指标 - 增强版"""
        with self.lock:
            if update_idx - self.last_update_idx >= self.update_frequency or self.last_update_idx == 0:
                # 计算基础指标
                mse = mean_squared_error(true_values, predictions)
                correlation = np.corrcoef(true_values, predictions)[0, 1] if len(true_values) > 1 else 0.0
                
                # 检测市场状态
                market_regime = self._detect_market_regime(true_values)
                
                # 自适应窗口调整
                self._update_adaptive_window(model_name, mse)
                
                # 计算稳定性分数
                stability_score = self._calculate_stability_score(model_name)
                
                # 计算性能趋势
                if len(self.model_performances[model_name]) > 0:
                    recent_mse = np.mean([p['mse'] for p in self.model_performances[model_name]])
                    mse_trend = (mse - recent_mse) / (recent_mse + 1e-8)
                    trend_strength = min(abs(mse_trend), 1.0)
                else:
                    mse_trend = 0.0
                    trend_strength = 0.0
                
                # 市场状态适应分数
                regime_bonus = {
                    'trending_up': 0.05,
                    'trending_down': 0.02,
                    'volatile': 0.01,  # 波动市场中稳定性更重要
                    'stable': 0.03,
                    'unknown': 0.0
                }.get(market_regime, 0.0)
                
                performance_score = {
                    'mse': mse,
                    'correlation': correlation if not np.isnan(correlation) else 0.0,
                    'stability_score': stability_score,
                    'trend_strength': trend_strength,
                    'trend_direction': 'improving' if mse_trend < 0 else 'deteriorating',
                    'market_regime': market_regime,
                    'regime_bonus': regime_bonus,
                    'timestamp': update_idx,
                    'sample_size': len(true_values)
                }
                
                # 保持窗口大小
                if len(self.model_performances[model_name]) >= self.window_size:
                    self.model_performances[model_name].popleft()
                
                self.model_performances[model_name].append(performance_score)
                self.last_update_idx = update_idx
                
    def get_model_weights(self, model_names: List[str], 
                         performance_boost: float = 0.15,
                         stability_boost: float = 0.12,
                         market_regime_aware: bool = True) -> Dict[str, float]:
        """基于历史性能和市场状态计算动态权重 - 增强版"""
        with self.lock:
            weights = {}
            total_weight = 0.0
            
            # 获取当前市场状态
            current_regime = 'unknown'
            if market_regime_aware and self.market_regime_history:
                current_regime = self.market_regime_history[-1]
            
            for model_name in model_names:
                if model_name not in self.model_performances:
                    # 新模型，渐进式权重分配
                    weights[model_name] = 0.8 / len(model_names)  # 降低新模型初始权重
                    total_weight += weights[model_name]
                    continue
                
                perf_history = list(self.model_performances[model_name])
                if not perf_history:
                    weights[model_name] = 0.8 / len(model_names)
                    total_weight += weights[model_name]
                    continue
                
                # 计算综合性能指标
                recent_perf = perf_history[-min(7, len(perf_history)):]  # 增加历史窗口
                
                # 多层性能指标
                avg_mse = np.mean([p['mse'] for p in recent_perf])
                avg_corr = np.mean([p['correlation'] for p in recent_perf])
                avg_stability = np.mean([p['stability_score'] for p in recent_perf])
                avg_trend = np.mean([p['trend_strength'] for p in recent_perf])
                
                # 时间衰减权重：最近数据更重要
                time_weights = np.exp(-np.arange(len(recent_perf)) * 0.1)
                time_weights /= time_weights.sum()
                
                weighted_mse = np.average([p['mse'] for p in recent_perf], weights=time_weights)
                weighted_stability = np.average([p['stability_score'] for p in recent_perf], weights=time_weights)
                weighted_trend = np.average([p['trend_strength'] for p in recent_perf], weights=time_weights)
                
                # 性能权重计算 - 增强版
                base_score = (1.0 / (1.0 + weighted_mse))  # MSE倒数
                correlation_score = max(0, avg_corr)  # 相关性(负值惩罚)
                stability_score = weighted_stability  # 稳定性
                trend_score = weighted_trend  # 趋势强度
                
                # 市场状态适应分数
                regime_bonus = 0.0
                if market_regime_aware and 'regime_bonus' in recent_perf[-1]:
                    regime_bonus = recent_perf[-1]['regime_bonus']
                
                # 综合评分：多因素权重
                performance_component = base_score * (1.0 + performance_boost)
                correlation_component = correlation_score * 0.5  # 相关性权重降低
                stability_component = stability_score * (1.0 + stability_boost)
                trend_component = trend_score * 0.3  # 趋势奖励
                regime_component = regime_bonus  # 市场适应奖励
                
                combined_score = (performance_component + correlation_component + 
                                stability_component + trend_component + regime_component)
                
                weights[model_name] = max(combined_score, 1e-8)
                total_weight += weights[model_name]
            
            # 归一化权重
            if total_weight > 0:
                for model_name in weights:
                    weights[model_name] /= total_weight
                    
                # 实施最小权重保护
                min_weight = 0.03  # 3%最小权重
                max_weight = 0.7   # 70%最大权重
                
                # 重新分配权重以满足约束
                excess_weights = {}
                for model_name, weight in weights.items():
                    if weight > max_weight:
                        excess_weights[model_name] = weight - max_weight
                        weights[model_name] = max_weight
                    elif weight < min_weight:
                        weights[model_name] = min_weight
                
                # 重新归一化
                total = sum(weights.values())
                for model_name in weights:
                    weights[model_name] /= total
            else:
                # 均匀分配
                equal_weight = 1.0 / len(model_names)
                weights = {name: equal_weight for name in model_names}
                
            return weights


class AdvancedConditionalWeightEngine:
    """增强条件化权重引擎 - 支持动态适应和多因子市场分析"""
    
    def __init__(self, market_lookback: int = 60, adaptation_rate: float = 0.05):
        self.market_lookback = market_lookback
        self.adaptation_rate = adaptation_rate
        self.market_states = deque(maxlen=market_lookback)
        self.state_transition_counts = defaultdict(int)
        self.state_performance_history = defaultdict(list)
        
        # 基础状态权重配置
        self.base_state_weights = {
            'trending_up': {'lightgbm': 0.45, 'xgboost': 0.35, 'catboost': 0.20},
            'trending_down': {'lightgbm': 0.25, 'xgboost': 0.35, 'catboost': 0.40},
            'volatile': {'lightgbm': 0.15, 'xgboost': 0.25, 'catboost': 0.60},
            'stable': {'lightgbm': 0.55, 'xgboost': 0.30, 'catboost': 0.15},
            'high_volatility': {'lightgbm': 0.10, 'xgboost': 0.20, 'catboost': 0.70},
            'low_volatility': {'lightgbm': 0.60, 'xgboost': 0.30, 'catboost': 0.10},
            'unknown': {'lightgbm': 1/3, 'xgboost': 1/3, 'catboost': 1/3}
        }
        
        # 自适应权重存储
        self.adaptive_state_weights = {}
        self.state_confidence = defaultdict(float)
        
    def _calculate_volatility_regime(self, returns: np.ndarray) -> str:
        """计算波动性状态"""
        if len(returns) < 5:
            return 'unknown'
            
        recent_vol = np.std(returns[-10:]) if len(returns) >= 10 else np.std(returns)
        long_term_vol = np.std(returns) if len(returns) > 10 else recent_vol
        
        # 相对波动性分析
        vol_ratio = recent_vol / (long_term_vol + 1e-8)
        
        if vol_ratio > 1.5:
            return 'high_volatility'
        elif vol_ratio < 0.7:
            return 'low_volatility'
        else:
            return 'stable'  # 默认到稳定状态
            
    def _update_adaptive_weights(self, current_state: str, model_performances: Dict[str, float]):
        """基于历史性能更新自适应权重"""
        if current_state not in self.base_state_weights:
            return
            
        # 更新状态转移计数
        if self.market_states:
            prev_state = self.market_states[-1]
            self.state_transition_counts[f"{prev_state}_to_{current_state}"] += 1
        
        # 记录状态性能
        for model_name, perf in model_performances.items():
            self.state_performance_history[f"{current_state}_{model_name}"].append(perf)
            
        # 计算自适应权重
        if len(self.state_performance_history[f"{current_state}_lightgbm"]) >= 3:
            adaptive_weights = {}
            total_perf = 0.0
            
            for model_name in ['lightgbm', 'xgboost', 'catboost']:
                history_key = f"{current_state}_{model_name}"
                if history_key in self.state_performance_history:
                    recent_perfs = self.state_performance_history[history_key][-5:]  # 最近5次
                    avg_perf = np.mean(recent_perfs)
                    # 性能越高权重越大（对于MSE，值越小越好）
                    perf_score = 1.0 / (1.0 + avg_perf)
                    adaptive_weights[model_name] = perf_score
                    total_perf += perf_score
            
            # 归一化
            if total_perf > 0:
                for model_name in adaptive_weights:
                    adaptive_weights[model_name] /= total_perf
                    
                # 混合基础权重和自适应权重
                base_weights = self.base_state_weights[current_state]
                mixed_weights = {}
                
                for model_name in base_weights:
                    if model_name in adaptive_weights:
                        # 70%自适应 + 30%基础
                        mixed_weights[model_name] = (0.7 * adaptive_weights[model_name] + 
                                                   0.3 * base_weights[model_name])
                    else:
                        mixed_weights[model_name] = base_weights[model_name]
                        
                # 重新归一化
                total = sum(mixed_weights.values())
                for model_name in mixed_weights:
                    mixed_weights[model_name] /= total
                    
                self.adaptive_state_weights[current_state] = mixed_weights
                
                # 更新状态置信度
                recent_transitions = sum(1 for key in self.state_transition_counts 
                                       if key.startswith(current_state))
                self.state_confidence[current_state] = min(1.0, recent_transitions / 10.0)
        
    def update_market_state(self, market_returns: np.ndarray, 
                          model_performances: Optional[Dict[str, float]] = None) -> str:
        """更新市场状态 - 增强版"""
        if len(market_returns) < 2:
            return 'unknown'
            
        # 多因子市场分析
        recent_returns = market_returns[-min(30, len(market_returns)):]
        long_term_returns = market_returns[-min(60, len(market_returns)):]
        
        # 计算多个市场特征
        short_trend = np.mean(recent_returns[-5:]) if len(recent_returns) >= 5 else np.mean(recent_returns)
        long_trend = np.mean(long_term_returns) if len(long_term_returns) > 5 else short_trend
        short_volatility = np.std(recent_returns[-10:]) if len(recent_returns) >= 10 else np.std(recent_returns)
        long_volatility = np.std(long_term_returns) if len(long_term_returns) > 10 else short_volatility
        
        # 趋势强度分析
        trend_consistency = np.mean(np.sign(recent_returns[-10:]) == np.sign(short_trend)) if len(recent_returns) >= 10 else 0.5
        
        # 动态阈值计算
        vol_percentile = np.percentile([np.std(market_returns[max(0, i-10):i]) for i in range(10, len(market_returns))], 75)
        vol_threshold = max(0.01, vol_percentile)
        
        # 多层次状态分类
        vol_regime = self._calculate_volatility_regime(market_returns)
        
        # 主要状态判断
        if short_volatility > vol_threshold * 2.5:
            primary_state = 'volatile'
        elif short_trend > vol_threshold * 4 and trend_consistency > 0.6:
            primary_state = 'trending_up'
        elif short_trend < -vol_threshold * 4 and trend_consistency > 0.6:
            primary_state = 'trending_down'
        else:
            primary_state = 'stable'
            
        # 状态细化：结合波动性状态
        if primary_state == 'stable':
            if vol_regime == 'high_volatility':
                final_state = 'high_volatility'
            elif vol_regime == 'low_volatility':
                final_state = 'low_volatility'
            else:
                final_state = 'stable'
        else:
            final_state = primary_state
            
        # 更新历史和自适应权重
        self.market_states.append(final_state)
        
        # 更新自适应权重（如果提供了模型性能）
        if model_performances:
            self._update_adaptive_weights(final_state, model_performances)
            
        return final_state
    
    def get_conditional_weights(self, current_state: str, 
                               confidence_threshold: float = 0.3) -> Dict[str, float]:
        """获取当前市场状态下的条件化权重 - 自适应版本"""
        # 优先使用自适应权重
        if current_state in self.adaptive_state_weights:
            confidence = self.state_confidence.get(current_state, 0.0)
            if confidence >= confidence_threshold:
                return self.adaptive_state_weights[current_state]
                
        # 回退到基础权重
        return self.base_state_weights.get(current_state, self.base_state_weights['unknown'])
    
    def get_state_transition_probability(self, next_state: str) -> Dict[str, float]:
        """获取状态转移概率"""
        if not self.market_states:
            return {state: 1.0/len(self.base_state_weights) for state in self.base_state_weights.keys()}
            
        current_state = self.market_states[-1]
        transition_key = f"{current_state}_to_{next_state}"
        total_transitions = sum(self.state_transition_counts.values())
        
        if total_transitions == 0:
            return {state: 1.0/len(self.base_state_weights) for state in self.base_state_weights.keys()}
            
        probabilities = {}
        for state in self.base_state_weights.keys():
            key = f"{current_state}_to_{state}"
            prob = self.state_transition_counts.get(key, 0) / total_transitions
            probabilities[state] = prob
            
        return probabilities


class DynamicWeightedEnsemble:
    """动态权重集成器，基于实时性能和市场状态调整权重"""
    
    def __init__(self, base_models: Sequence, initial_weights: Optional[Sequence[float]] = None,
                 performance_window: int = 100, conditional_weighting: bool = True,
                 weight_smoothing: float = 0.1, min_weight_threshold: float = 0.05):
        self.base_models = list(base_models)
        if not self.base_models:
            raise ValueError("至少需要一个基础模型用于集成")
            
        self.model_names = [f"model_{i}" for i in range(len(self.base_models))]
        self.initial_weights = self._normalize_weights(initial_weights)
        self.performance_window = performance_window
        self.conditional_weighting = conditional_weighting
        self.weight_smoothing = max(0.01, min(0.5, weight_smoothing))
        self.min_weight_threshold = min_weight_threshold
        
        # 初始化组件
        self.performance_monitor = AdvancedModelPerformanceMonitor(window_size=performance_window)
        self.conditional_engine = AdvancedConditionalWeightEngine()
        self.current_weights = self.initial_weights.copy()
        self.prediction_count = 0
        self.model_failures = defaultdict(int)
        self.last_successful_predictions = defaultdict(int)
        self.weight_history = []  # 权重变化历史
        
    def _normalize_weights(self, weights: Optional[Sequence[float]]) -> np.ndarray:
        if weights is None:
            arr = np.ones(len(self.base_models), dtype=float) / len(self.base_models)
        else:
            arr = np.asarray(list(weights), dtype=float)
            if arr.shape[0] != len(self.base_models):
                raise ValueError("权重数量必须与基础模型数量一致")
        total = arr.sum()
        return arr / total if total > 0 else np.ones(len(self.base_models)) / len(self.base_models)
    
    def _get_fallback_weights(self) -> np.ndarray:
        """获取回退权重（故障模型惩罚）"""
        fallback_weights = np.ones(len(self.base_models), dtype=float)
        current_time = self.prediction_count
        
        for i, model_name in enumerate(self.model_names):
            last_success = self.last_successful_predictions[model_name]
            failure_count = self.model_failures[model_name]
            
            # 如果模型长时间无成功预测或失败次数过多，减少权重
            if (current_time - last_success) > 50:  # 50次预测无成功
                fallback_weights[i] *= 0.1
            elif failure_count > 5:  # 失败超过5次
                fallback_weights[i] *= 0.2
            else:
                # 基础权重衰减
                time_decay = np.exp(-(current_time - last_success) / 100.0)
                fallback_weights[i] *= time_decay
        
        # 归一化
        total = fallback_weights.sum()
        return fallback_weights / total if total > 0 else np.ones(len(self.base_models)) / len(self.base_models)
    
    def _update_weights(self, predictions: np.ndarray, true_values: Optional[np.ndarray] = None, 
                       model_performances: Optional[Dict[str, float]] = None):
        """更新模型权重 - 增强版"""
        if true_values is None:
            # 无真实值时使用平滑回退
            fallback_weights = self._get_fallback_weights()
            smoothing_factor = self.weight_smoothing
        else:
            # 基于性能更新权重
            with ThreadPoolExecutor(max_workers=1) as executor:
                futures = []
                for i, (model, model_name) in enumerate(zip(self.base_models, self.model_names)):
                    pred_slice = predictions[:, i] if predictions.ndim > 1 else predictions
                    future = executor.submit(
                        self.performance_monitor.update_performance,
                        model_name, true_values, pred_slice,
                        self.prediction_count
                    )
                    futures.append(future)
                
                # 等待所有更新完成
                for future in futures:
                    try:
                        future.result(timeout=5.0)  # 5秒超时
                    except Exception as e:
                        print(f"⚠️ 性能更新失败: {e}")
            
            # 获取性能权重 - 增强版
            perf_weights_dict = self.performance_monitor.get_model_weights(
                self.model_names, 
                performance_boost=0.15,
                stability_boost=0.12,
                market_regime_aware=True
            )
            perf_weights = np.array([perf_weights_dict[name] for name in self.model_names])
            
            # 如果启用条件化权重，混合市场状态权重
            if self.conditional_weighting and len(true_values) > 0:
                # 更新市场状态并获取条件化权重
                market_state = self.conditional_engine.update_market_state(true_values, model_performances)
                conditional_weights_dict = self.conditional_engine.get_conditional_weights(market_state)
                conditional_weights = np.array([conditional_weights_dict.get(name.split('_')[1], 1/3) 
                                              for name in self.model_names])
                
                # 动态混合权重：60%性能权重 + 40%条件化权重（增加条件化权重）
                mixed_weights = 0.6 * perf_weights + 0.4 * conditional_weights
                fallback_weights = mixed_weights
            else:
                fallback_weights = perf_weights
            
            smoothing_factor = self.weight_smoothing * 0.8  # 略微增加更新速度
        
        # 应用权重平滑
        self.current_weights = (1 - smoothing_factor) * self.current_weights + smoothing_factor * fallback_weights
        
        # 应用最小权重阈值 - 增强保护
        min_total = self.min_weight_threshold * len(self.base_models)
        if self.current_weights.sum() < min_total:
            self.current_weights = np.ones(len(self.base_models)) / len(self.base_models)
        
        # 实施最大权重约束防止过拟合
        max_weight = 0.6  # 60%最大权重
        for i in range(len(self.current_weights)):
            if self.current_weights[i] > max_weight:
                excess = self.current_weights[i] - max_weight
                self.current_weights[i] = max_weight
                # 将超出部分平均分配给其他模型
                other_indices = [j for j in range(len(self.current_weights)) if j != i]
                if other_indices:
                    self.current_weights[other_indices] += excess / len(other_indices)
        
        # 确保权重和为1
        self.current_weights /= self.current_weights.sum()
        
        # 记录权重变化用于分析
        self.weight_history.append(self.current_weights.copy())
        if len(self.weight_history) > 50:  # 保持最近50次权重变化
            self.weight_history.pop(0)
    
    def fit(self, X, y):
        """训练所有基础模型"""
        for model in self.base_models:
            try:
                model.fit(X, y)
            except Exception as e:
                print(f"⚠️ 模型训练失败: {e}")
                # 继续训练其他模型
                continue
        return self
    
    def predict(self, X, return_model_predictions: bool = False):
        """预测"""
        predictions = []
        model_success_flags = []
        
        # 收集所有模型预测
        for i, (model, model_name) in enumerate(zip(self.base_models, self.model_names)):
            try:
                pred = model.predict(X)
                if len(pred) != len(X):
                    raise ValueError(f"预测长度不匹配: {len(pred)} vs {len(X)}")
                
                predictions.append(pred)
                model_success_flags.append(True)
                self.last_successful_predictions[model_name] = self.prediction_count
                
            except Exception as e:
                print(f"⚠️ 模型 {model_name} 预测失败: {e}")
                model_failures[model_name] += 1
                model_success_flags.append(False)
                
                # 使用回退预测（简单平均值）
                if predictions:
                    fallback_pred = np.mean(predictions, axis=0)  # 使用已有预测的平均值
                else:
                    fallback_pred = np.ones(len(X))  # 全1向量作为最保守的回退
                predictions.append(fallback_pred)
        
        if not predictions:
            raise RuntimeError("所有模型预测都失败了")
        
        # 转换为numpy数组
        all_predictions = np.array(predictions).T  # 转置：行=样本，列=模型
        
        # 更新权重
        if self.prediction_count % 5 == 0:  # 每5次预测更新一次权重（更频繁）
            # 获取模型性能指标用于市场状态检测
            model_perfs = {}
            for i, (model, model_name) in enumerate(zip(self.base_models, self.model_names)):
                if model_success_flags[i] and i < len(all_predictions[0]):
                    try:
                        # 简化的性能指标：预测变异性
                        pred_var = np.var(all_predictions[:, i])
                        model_perfs[model_name] = pred_var
                    except:
                        model_perfs[model_name] = 0.1
            
            self._update_weights(all_predictions, None, model_perfs)  # 使用性能指标更新市场状态
        
        # 计算加权平均预测
        current_weights = self.current_weights[model_success_flags]
        if len(current_weights) != sum(model_success_flags):
            # 权重数量不匹配，调整为成功模型数量
            current_weights = np.ones(sum(model_success_flags)) / sum(model_success_flags)
        
        weighted_prediction = np.average(all_predictions[:, model_success_flags], 
                                       axis=1, weights=current_weights)
        
        self.prediction_count += 1
        
        if return_model_predictions:
            return weighted_prediction, all_predictions
        else:
            return weighted_prediction


class StackingEnsemble:
    """Stacking集成器，使用元学习器组合基础模型预测"""
    
    def __init__(self, base_models: Sequence, meta_model: Optional[Any] = None,
                 cv_folds: int = 3, use_features_in_secondary: bool = False,
                 stack_method: str = 'predict_proba'):
        self.base_models = list(base_models)
        self.cv_folds = max(2, cv_folds)
        self.use_features_in_secondary = use_features_in_secondary
        self.stack_method = stack_method
        
        # 初始化元学习器
        if meta_model is None:
            try:
                from sklearn.linear_model import Ridge
                self.meta_model = Ridge(alpha=1.0, random_state=42)
            except ImportError:
                # 简单线性回归作为回退
                self.meta_model = self._create_simple_meta_model()
        else:
            self.meta_model = meta_model
            
        self.base_model_names = [f"stacker_base_{i}" for i in range(len(self.base_models))]
        self.is_fitted = False
        self.feature_columns = None
        
    def _create_simple_meta_model(self):
        """创建简单的元学习器作为回退"""
        class SimpleMetaModel:
            def __init__(self):
                self.weights = None
                self.intercept = None
                
            def fit(self, X, y):
                X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
                try:
                    self.weights, self.intercept = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0:2]
                except:
                    # 奇异值分解回退
                    self.weights = np.zeros(X.shape[1])
                    self.intercept = np.mean(y)
                    
            def predict(self, X):
                X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
                return X_with_intercept @ np.concatenate([[self.intercept], self.weights])
                
        return SimpleMetaModel()
    
    def _get_out_of_fold_predictions(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """获取OOF预测"""
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        oof_predictions = np.zeros((len(X), len(self.base_models)))
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            for i, (base_model, model_name) in enumerate(zip(self.base_models, self.base_model_names)):
                try:
                    # 重新创建模型以避免数据泄漏
                    import copy
                    temp_model = copy.deepcopy(base_model)
                    temp_model.fit(X_train_fold, y_train_fold)
                    
                    # 预测验证集
                    pred = temp_model.predict(X_val_fold)
                    oof_predictions[val_idx, i] = pred.flatten() if hasattr(pred, 'flatten') else pred
                    
                except Exception as e:
                    print(f"⚠️ 折叠 {fold} 模型 {i} OOF预测失败: {e}")
                    # 使用基线预测作为回退
                    oof_predictions[val_idx, i] = np.mean(y_train_fold) if len(y_train_fold) > 0 else 0.0
                    
        return oof_predictions
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """训练Stacking集成器"""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
            
        self.feature_columns = X.columns.tolist()
        
        # 第一阶段：训练基础模型并获取OOF预测
        oof_predictions = self._get_out_of_fold_predictions(X, y)
        
        # 第二阶段：准备元学习器的特征
        if self.use_features_in_secondary:
            # 使用原始特征 + OOF预测
            meta_features = np.column_stack([X.values, oof_predictions])
        else:
            # 只使用OOF预测
            meta_features = oof_predictions
            
        # 训练元学习器
        try:
            self.meta_model.fit(meta_features, y.values)
            self.is_fitted = True
        except Exception as e:
            print(f"⚠️ 元学习器训练失败: {e}")
            # 使用简单平均作为回退
            self.is_fitted = False
            
        # 最后，使用全部数据重新训练基础模型
        for base_model in self.base_models:
            try:
                base_model.fit(X, y)
            except Exception as e:
                print(f"⚠️ 基础模型重新训练失败: {e}")
                
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            # 如果元学习器未训练，使用简单平均
            predictions = np.zeros((len(X), len(self.base_models)))
            for i, base_model in enumerate(self.base_models):
                try:
                    predictions[:, i] = base_model.predict(X)
                except:
                    predictions[:, i] = 0.0  # 回退预测
            return np.mean(predictions, axis=1)
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        # 基础模型预测
        base_predictions = np.zeros((len(X), len(self.base_models)))
        for i, base_model in enumerate(self.base_models):
            try:
                base_predictions[:, i] = base_model.predict(X)
            except Exception as e:
                print(f"⚠️ 基础模型 {i} 预测失败: {e}")
                base_predictions[:, i] = 0.0  # 回退预测
        
        # 元学习器预测
        if self.use_features_in_secondary:
            meta_features = np.column_stack([X.values, base_predictions])
        else:
            meta_features = base_predictions
            
        try:
            final_predictions = self.meta_model.predict(meta_features)
            return final_predictions.flatten() if hasattr(final_predictions, 'flatten') else final_predictions
        except Exception as e:
            print(f"⚠️ 元学习器预测失败，使用基础模型平均: {e}")
            return np.mean(base_predictions, axis=1)


class AdversarialEnsemble:
    """对抗性集成器 - 使用对抗样本增强模型鲁棒性"""
    
    def __init__(self, base_models: Sequence, adversarial_ratio: float = 0.1,
                 noise_std: float = 0.01, robustness_weight: float = 0.2):
        self.base_models = list(base_models)
        self.adversarial_ratio = adversarial_ratio
        self.noise_std = noise_std
        self.robustness_weight = robustness_weight
        
        self.model_robustness_scores = defaultdict(float)
        self.robustness_history = defaultdict(list)
        
    def _generate_adversarial_examples(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """生成对抗样本"""
        n_adversarial = int(len(X) * self.adversarial_ratio)
        if n_adversarial == 0:
            return X, y
            
        # 随机选择样本
        indices = np.random.choice(len(X), size=n_adversarial, replace=False)
        
        # 处理不同类型的X（numpy array或DataFrame）
        if isinstance(X, pd.DataFrame):
            X_adv = X.iloc[indices].copy().values  # 转换为numpy array
        else:
            X_adv = X[indices].copy()
        
        # 添加小扰动（快速梯度符号法简化版）
        for i in range(len(X_adv)):
            noise = np.random.normal(0, self.noise_std, X_adv[i].shape)
            X_adv[i] += noise
            
        # 处理y的类型
        if isinstance(y, pd.Series):
            y_adv = y.iloc[indices].values
        else:
            y_adv = y[indices]
        
        return X_adv, y_adv
    
    def _evaluate_robustness(self, model_idx: int, X_clean: np.ndarray, y_clean: np.ndarray,
                           X_adv: np.ndarray, y_adv: np.ndarray) -> float:
        """评估模型鲁棒性"""
        try:
            model = self.base_models[model_idx]
            
            # 清洁样本性能
            pred_clean = model.predict(X_clean)
            clean_mse = mean_squared_error(y_clean, pred_clean)
            
            # 对抗样本性能
            if len(X_adv) > 0:
                pred_adv = model.predict(X_adv)
                adv_mse = mean_squared_error(y_adv, pred_adv)
                
                # 鲁棒性分数：对抗损失相对较小表示更好的鲁棒性
                robustness = clean_mse / (adv_mse + 1e-8)
                return min(robustness, 1.0)  # 限制在[0,1]
            else:
                return 1.0
                
        except Exception as e:
            print(f"⚠️ 鲁棒性评估失败: {e}")
            return 0.5
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练对抗性集成器"""
        # 生成对抗样本
        X_adv, y_adv = self._generate_adversarial_examples(X, y)
        
        # 训练所有基础模型
        for i, model in enumerate(self.base_models):
            try:
                model.fit(X, y)
                
                # 评估鲁棒性
                if len(X_adv) > 0:
                    robustness = self._evaluate_robustness(i, X[:min(100, len(X))], y[:min(100, len(y))],
                                                         X_adv[:min(50, len(X_adv))], y_adv[:min(50, len(y_adv))])
                    self.model_robustness_scores[i] = robustness
                    self.robustness_history[i].append(robustness)
                    
            except Exception as e:
                print(f"⚠️ 模型 {i} 训练失败: {e}")
                self.model_robustness_scores[i] = 0.5
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        all_predictions = []
        
        for i, model in enumerate(self.base_models):
            try:
                pred = model.predict(X)
                all_predictions.append(pred.flatten() if hasattr(pred, 'flatten') else pred)
            except Exception as e:
                print(f"⚠️ 模型 {i} 预测失败: {e}")
                all_predictions.append(np.zeros(len(X)))
        
        if not all_predictions:
            return np.zeros(len(X))
            
        predictions_array = np.array(all_predictions).T
        
        # 基于鲁棒性计算权重
        robustness_scores = [self.model_robustness_scores.get(i, 0.5) for i in range(len(self.base_models))]
        total_robustness = sum(robustness_scores)
        
        if total_robustness > 0:
            # 60%性能权重 + 40%鲁棒性权重（简化版）
            equal_weight = 1.0 / len(self.base_models)
            robust_weights = [score / total_robustness for score in robustness_scores]
            
            # 混合权重
            final_weights = [0.6 * equal_weight + 0.4 * robust_weight 
                           for robust_weight in robust_weights]
        else:
            final_weights = [1.0 / len(self.base_models)] * len(self.base_models)
        
        return np.average(predictions_array, axis=1, weights=final_weights)


class MultiLevelEnsemble:
    """多层次集成器 - 分层组合不同类型的模型"""
    
    def __init__(self, level1_models: Sequence, level2_models: Sequence, 
                 meta_ensemble: Optional[Any] = None, blending_ratio: float = 0.7):
        self.level1_models = list(level1_models)
        self.level2_models = list(level2_models)
        self.meta_ensemble = meta_ensemble
        self.blending_ratio = blending_ratio
        
        # 层次权重
        self.level1_weights = np.ones(len(level1_models)) / len(level1_models)
        self.level2_weights = np.ones(len(level2_models)) / len(level2_models)
        self.meta_weights = np.array([blending_ratio, 1 - blending_ratio])
        
    def _create_meta_features(self, level1_preds: np.ndarray, level2_preds: np.ndarray) -> np.ndarray:
        """创建元学习器特征"""
        # 组合两个层次的预测
        meta_features = np.column_stack([
            level1_preds.mean(axis=1),  # Level1平均值
            level1_preds.std(axis=1),   # Level1标准差
            level2_preds.mean(axis=1),  # Level2平均值
            level2_preds.std(axis=1),   # Level2标准差
            level1_preds.max(axis=1),   # Level1最大值
            level1_preds.min(axis=1),   # Level1最小值
            level2_preds.max(axis=1),   # Level2最大值
            level2_preds.min(axis=1)    # Level2最小值
        ])
        return meta_features
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练多层次集成器"""
        # 训练Level1模型
        for model in self.level1_models:
            try:
                model.fit(X, y)
            except Exception as e:
                print(f"⚠️ Level1模型训练失败: {e}")
                
        # 训练Level2模型
        for model in self.level2_models:
            try:
                model.fit(X, y)
            except Exception as e:
                print(f"⚠️ Level2模型训练失败: {e}")
        
        # 训练元集成器（如果提供）
        if self.meta_ensemble is not None:
            try:
                # 获取Level1和Level2预测
                level1_preds = np.array([model.predict(X) for model in self.level1_models]).T
                level2_preds = np.array([model.predict(X) for model in self.level2_models]).T
                
                # 创建元特征
                meta_features = self._create_meta_features(level1_preds, level2_preds)
                
                # 训练元学习器
                self.meta_ensemble.fit(meta_features, y)
                
            except Exception as e:
                print(f"⚠️ 元学习器训练失败: {e}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        # Level1预测
        level1_preds = np.array([model.predict(X) for model in self.level1_models]).T
        level1_final = np.average(level1_preds, axis=1, weights=self.level1_weights)
        
        # Level2预测
        level2_preds = np.array([model.predict(X) for model in self.level2_models]).T
        level2_final = np.average(level2_preds, axis=1, weights=self.level2_weights)
        
        # 元集成器预测
        if self.meta_ensemble is not None:
            try:
                meta_features = self._create_meta_features(level1_preds, level2_preds)
                meta_pred = self.meta_ensemble.predict(meta_features)
                return meta_pred.flatten() if hasattr(meta_pred, 'flatten') else meta_pred
            except Exception as e:
                print(f"⚠️ 元学习器预测失败: {e}")
        
        # 简单混合
        final_pred = (self.meta_weights[0] * level1_final + 
                     self.meta_weights[1] * level2_final)
        
        return final_pred


class RiskAwareEnsemble:
    """风险感知集成器 - 增强版，结合预测不确定性和风险约束"""
    
    def __init__(self, base_models: Sequence, risk_models: Optional[Sequence] = None,
                 volatility_constraint: float = 1.2, uncertainty_threshold: float = 0.1,
                 risk_parity: bool = True, dynamic_risk_adjustment: bool = True):
        self.base_models = list(base_models)
        self.risk_models = risk_models or base_models
        self.volatility_constraint = volatility_constraint
        self.uncertainty_threshold = uncertainty_threshold
        self.risk_parity = risk_parity
        self.dynamic_risk_adjustment = dynamic_risk_adjustment
        
        self.model_uncertainties = defaultdict(list)
        self.risk_adjusted_weights = None
        self.risk_history = defaultdict(list)
        self.current_risk_regime = 'unknown'
        
        # 风险适应参数
        self.risk_lookback = 20
        self.volatility_ma_length = 10
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """训练风险感知集成器"""
        # 训练所有基础模型
        for model in self.base_models:
            try:
                model.fit(X, y)
            except Exception as e:
                print(f"⚠️ 模型训练失败: {e}")
                
        # 训练风险模型（如果不同于基础模型）
        if self.risk_models != self.base_models:
            for model in self.risk_models:
                try:
                    model.fit(X, y)
                except Exception as e:
                    print(f"⚠️ 风险模型训练失败: {e}")
        
        return self
    
    def predict(self, X: pd.DataFrame, return_uncertainty: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """预测，包含不确定性估计"""
        all_predictions = []
        all_uncertainties = []
        
        # 获取所有模型预测
        for i, (base_model, risk_model) in enumerate(zip(self.base_models, self.risk_models)):
            try:
                pred = base_model.predict(X)
                uncertainty = self._estimate_uncertainty(i, X, risk_model)
                
                all_predictions.append(pred.flatten() if hasattr(pred, 'flatten') else pred)
                all_uncertainties.append(uncertainty)
                
            except Exception as e:
                print(f"⚠️ 模型 {i} 预测失败: {e}")
                all_predictions.append(np.zeros(len(X)))
                all_uncertainties.append(np.ones(len(X)) * 0.2)
        
        # 转换为数组
        predictions_array = np.array(all_predictions).T  # 形状: (样本数, 模型数)
        uncertainties_array = np.array(all_uncertainties).T
        
        if self.risk_parity:
            # 风险平价权重
            avg_uncertainty = np.mean(uncertainties_array, axis=0)
            risk_weights = 1.0 / (avg_uncertainty + 1e-8)
            risk_weights /= risk_weights.sum()
        else:
            # 均匀权重
            risk_weights = np.ones(len(self.base_models)) / len(self.base_models)
        
        # 计算风险调整预测
        final_predictions = np.average(predictions_array, axis=1, weights=risk_weights)
        
        if return_uncertainty:
            avg_uncertainty = np.average(uncertainties_array, axis=1, weights=risk_weights)
            return final_predictions, avg_uncertainty
        else:
            return final_predictions
        
    def _estimate_uncertainty(self, model_idx: int, X: pd.DataFrame, 
                             base_model: Any) -> np.ndarray:
        """估计模型预测不确定性"""
        try:
            # 使用bootstrap或ensemble方法来估计不确定性
            bootstrap_predictions = []
            n_bootstrap = 10
            
            for _ in range(n_bootstrap):
                # Bootstrap采样
                indices = np.random.choice(len(X), size=len(X), replace=True)
                X_bootstrap = X.iloc[indices]
                
                try:
                    # 重新训练模型（简化版本）
                    pred = base_model.predict(X_bootstrap[:min(100, len(X_bootstrap))])
                    bootstrap_predictions.append(pred)
                except:
                    continue
            
            if bootstrap_predictions:
                # 计算预测方差作为不确定性
                bootstrap_array = np.array(bootstrap_predictions)
                uncertainty = np.std(bootstrap_array, axis=0)
                return uncertainty
            else:
                # 回退：使用固定不确定性
                return np.ones(len(X)) * 0.1
                
        except Exception as e:
            print(f"⚠️ 不确定性估计失败: {e}")
            return np.ones(len(X)) * 0.1

    def _calculate_risk_contribution(self, predictions: np.ndarray, 
                                   uncertainties: np.ndarray, 
                                   risk_regime: str = 'normal_risk') -> np.ndarray:
        """计算风险贡献 - 增强版"""
        if uncertainties is None:
            uncertainties = np.ones_like(predictions) * 0.1
            
        # 基础风险调整
        base_risk_factor = 1.0 / (1.0 + uncertainties)
        
        # 风险状态调节
        regime_adjustments = {
            'high_risk': 0.6,      # 高风险时降低权重
            'medium_risk': 0.8,    # 中等风险时适度降低
            'normal_risk': 1.0,    # 正常风险
            'low_risk': 1.2        # 低风险时增加权重
        }
        
        regime_factor = regime_adjustments.get(risk_regime, 1.0)
        
        # 动态风险调整
        if self.dynamic_risk_adjustment:
            # 基于预测幅度的调整
            pred_magnitude = np.abs(predictions)
            magnitude_factor = 1.0 / (1.0 + pred_magnitude)
            
            # 综合风险调整
            final_risk_factor = base_risk_factor * regime_factor * magnitude_factor
        else:
            final_risk_factor = base_risk_factor * regime_factor
            
        return final_risk_factor
    
    def _update_risk_history(self, model_idx: int, risk_score: float):
        """更新风险历史记录"""
        self.risk_history[model_idx].append(risk_score)
        
        # 保持历史长度
        if len(self.risk_history[model_idx]) > self.risk_lookback:
            self.risk_history[model_idx] = self.risk_history[model_idx][-self.risk_lookback:]


class RealTimePerformanceMonitor:
    """实时性能监控和自动故障检测系统"""
    
    def __init__(self, health_check_interval: int = 10, 
                 failure_threshold: float = 0.1,
                 recovery_threshold: float = 0.05,
                 max_failures: int = 3):
        self.health_check_interval = health_check_interval
        self.failure_threshold = failure_threshold
        self.recovery_threshold = recovery_threshold
        self.max_failures = max_failures
        
        self.model_health_scores = defaultdict(float)
        self.failure_counts = defaultdict(int)
        self.last_health_check = 0
        self.performance_baseline = {}
        self.recovery_history = defaultdict(list)
        
    def assess_model_health(self, model_name: str, recent_performance: float, 
                          baseline_performance: Optional[float] = None) -> str:
        """评估模型健康状态"""
        if baseline_performance is None:
            baseline_performance = self.performance_baseline.get(model_name, recent_performance)
        else:
            self.performance_baseline[model_name] = baseline_performance
        
        # 计算性能下降幅度
        performance_drop = (recent_performance - baseline_performance) / (baseline_performance + 1e-8)
        
        # 健康状态评估
        if performance_drop > self.failure_threshold:
            status = 'failing'
            self.failure_counts[model_name] += 1
        elif performance_drop < -self.recovery_threshold:
            # 性能显著改善，可能从故障中恢复
            if self.failure_counts[model_name] > 0:
                self.failure_counts[model_name] -= 1
                self.recovery_history[model_name].append(datetime.now())
            status = 'recovering' if self.failure_counts[model_name] > 0 else 'healthy'
        else:
            status = 'healthy'
            self.failure_counts[model_name] = max(0, self.failure_counts[model_name] - 0.1)  # 缓慢恢复
        
        # 计算健康分数
        if status == 'failing':
            health_score = max(0.0, 1.0 - abs(performance_drop) * 2)
        elif status == 'recovering':
            health_score = 0.7
        else:
            health_score = 1.0
            
        self.model_health_scores[model_name] = health_score
        return status
    
    def get_replacement_candidates(self, current_models: List[str]) -> List[str]:
        """获取可能的模型替换候选"""
        failing_models = []
        for model_name in current_models:
            if (self.failure_counts[model_name] >= self.max_failures or 
                self.model_health_scores.get(model_name, 1.0) < 0.3):
                failing_models.append(model_name)
        return failing_models
    
    def get_health_summary(self) -> Dict[str, Any]:
        """获取健康状态总结"""
        summary = {
            'total_models': len(self.model_health_scores),
            'healthy_models': sum(1 for score in self.model_health_scores.values() if score > 0.7),
            'failing_models': sum(1 for score in self.model_health_scores.values() if score < 0.3),
            'model_details': dict(self.model_health_scores),
            'failure_counts': dict(self.failure_counts),
            'recovered_models': len(self.recovery_history)
        }
        return summary
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """训练风险感知集成器"""
        # 训练所有基础模型
        for model in self.base_models:
            try:
                model.fit(X, y)
            except Exception as e:
                print(f"⚠️ 模型训练失败: {e}")
                
        # 训练风险模型（如果不同于基础模型）
        if self.risk_models != self.base_models:
            for model in self.risk_models:
                try:
                    model.fit(X, y)
                except Exception as e:
                    print(f"⚠️ 风险模型训练失败: {e}")
        
        return self
    
    def predict(self, X: pd.DataFrame, return_uncertainty: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """预测，包含不确定性估计"""
        all_predictions = []
        all_uncertainties = []
        
        # 获取所有模型预测
        for i, (base_model, risk_model) in enumerate(zip(self.base_models, self.risk_models)):
            try:
                pred = base_model.predict(X)
                uncertainty = self._estimate_uncertainty(i, X, risk_model)
                
                all_predictions.append(pred.flatten() if hasattr(pred, 'flatten') else pred)
                all_uncertainties.append(uncertainty)
                
            except Exception as e:
                print(f"⚠️ 模型 {i} 预测失败: {e}")
                all_predictions.append(np.zeros(len(X)))
                all_uncertainties.append(np.ones(len(X)) * 0.2)
        
        # 转换为数组
        predictions_array = np.array(all_predictions).T  # 形状: (样本数, 模型数)
        uncertainties_array = np.array(all_uncertainties).T
        
        if self.risk_parity:
            # 风险平价权重
            avg_uncertainty = np.mean(uncertainties_array, axis=0)
            risk_weights = 1.0 / (avg_uncertainty + 1e-8)
            risk_weights /= risk_weights.sum()
        else:
            # 均匀权重
            risk_weights = np.ones(len(self.base_models)) / len(self.base_models)
        
        # 计算风险调整预测
        final_predictions = np.average(predictions_array, axis=1, weights=risk_weights)
        
        if return_uncertainty:
            avg_uncertainty = np.average(uncertainties_array, axis=1, weights=risk_weights)
            return final_predictions, avg_uncertainty
        else:
            return final_predictions


class AveragingEnsemble:
    """简单均值集成器，支持自定义权重"""

    def __init__(self, base_models: Sequence, weights: Sequence[float] | None = None):
        self.base_models = list(base_models)
        if not self.base_models:
            raise ValueError("至少需要一个基础模型用于集成")
        self.weights = self._normalize_weights(weights)

    def _normalize_weights(self, weights: Sequence[float] | None) -> np.ndarray:
        if weights is None:
            arr = np.ones(len(self.base_models), dtype=float)
        else:
            arr = np.asarray(list(weights), dtype=float)
            if arr.shape[0] != len(self.base_models):
                raise ValueError("权重数量必须与基础模型数量一致")
        total = arr.sum()
        if total <= 0:
            arr = np.ones(len(self.base_models), dtype=float)
            total = arr.sum()
        return arr / total

    def fit(self, X, y):
        for model in self.base_models:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.vstack([model.predict(X) for model in self.base_models])
        return np.average(predictions, axis=0, weights=self.weights)


class HullModel:
    """Hull Tactical 预测模型 - 增强版支持高级集成策略"""
    
    def __init__(
        self,
        model_type: str = "baseline",
        model_params: Optional[Dict[str, Any]] = None,
        *,
        random_state: Optional[int] = 42,
        auto_validation_fraction: float = 0.1,
        enable_early_stopping: bool = True,
        ensemble_config: Optional[Dict[str, Any]] = None,
    ):
        self.model_type = model_type.lower()
        params = dict(model_params or {})
        self.fit_params = params.pop("fit_params", {})
        self.model_params = params
        self.ensemble_config = ensemble_config or {}
        self.model = None
        self.feature_columns = None
        self.random_state = random_state
        self.auto_validation_fraction = auto_validation_fraction
        self.enable_early_stopping = enable_early_stopping
        self.min_early_stopping_rows = 500

    def _build_model(self):
        if self.model_type == "baseline":
            return create_baseline_model(random_state=self.random_state, **self.model_params)
        if self.model_type == "lightgbm":
            return create_lightgbm_model(random_state=self.random_state, **self.model_params)
        if self.model_type == "xgboost":
            return create_xgboost_model(random_state=self.random_state, **self.model_params)
        if self.model_type == "catboost":
            return create_catboost_model(random_state=self.random_state, **self.model_params)
        if self.model_type == "ensemble":
            return self._build_ensemble_model()
        if self.model_type == "dynamic_weighted_ensemble":
            return self._build_dynamic_weighted_ensemble()
        if self.model_type == "stacking_ensemble":
            return self._build_stacking_ensemble()
        if self.model_type == "risk_aware_ensemble":
            return self._build_risk_aware_ensemble()
        raise ValueError(f"Unknown model type: {self.model_type}")

    def _build_ensemble_model(self):
        """构建基础集成模型"""
        ensemble_params = dict(self.model_params)
        weight_cfg = ensemble_params.pop("weights", None)
        base_models = [
            create_lightgbm_model(random_state=self.random_state, **ensemble_params.get("lightgbm", {})),
            create_xgboost_model(random_state=self.random_state, **ensemble_params.get("xgboost", {})),
            create_catboost_model(random_state=self.random_state, **ensemble_params.get("catboost", {})),
        ]
        weights = self._resolve_ensemble_weights(weight_cfg, len(base_models))
        return AveragingEnsemble(base_models, weights=weights)
    
    def _build_dynamic_weighted_ensemble(self):
        """构建动态权重集成模型"""
        ensemble_params = dict(self.model_params)
        base_models = [
            create_lightgbm_model(random_state=self.random_state, **ensemble_params.get("lightgbm", {})),
            create_xgboost_model(random_state=self.random_state, **ensemble_params.get("xgboost", {})),
            create_catboost_model(random_state=self.random_state, **ensemble_params.get("catboost", {})),
        ]
        
        # 动态权重配置
        dyn_config = {
            'performance_window': self.ensemble_config.get('performance_window', 100),
            'conditional_weighting': self.ensemble_config.get('conditional_weighting', True),
            'weight_smoothing': self.ensemble_config.get('weight_smoothing', 0.1),
            'min_weight_threshold': self.ensemble_config.get('min_weight_threshold', 0.05),
        }
        
        return DynamicWeightedEnsemble(base_models, **dyn_config)
    
    def _build_stacking_ensemble(self):
        """构建Stacking集成模型"""
        ensemble_params = dict(self.model_params)
        base_models = [
            create_lightgbm_model(random_state=self.random_state, **ensemble_params.get("lightgbm", {})),
            create_xgboost_model(random_state=self.random_state, **ensemble_params.get("xgboost", {})),
            create_catboost_model(random_state=self.random_state, **ensemble_params.get("catboost", {})),
        ]
        
        # Stacking配置
        stack_config = {
            'cv_folds': self.ensemble_config.get('cv_folds', 3),
            'use_features_in_secondary': self.ensemble_config.get('use_features_in_secondary', False),
            'stack_method': self.ensemble_config.get('stack_method', 'predict'),
        }
        
        return StackingEnsemble(base_models, **stack_config)
    
    def _build_risk_aware_ensemble(self):
        """构建风险感知集成模型"""
        ensemble_params = dict(self.model_params)
        base_models = [
            create_lightgbm_model(random_state=self.random_state, **ensemble_params.get("lightgbm", {})),
            create_xgboost_model(random_state=self.random_state, **ensemble_params.get("xgboost", {})),
            create_catboost_model(random_state=self.random_state, **ensemble_params.get("catboost", {})),
        ]
        
        # 风险感知配置
        risk_config = {
            'volatility_constraint': self.ensemble_config.get('volatility_constraint', 1.2),
            'uncertainty_threshold': self.ensemble_config.get('uncertainty_threshold', 0.1),
            'risk_parity': self.ensemble_config.get('risk_parity', True),
        }
        
        return RiskAwareEnsemble(base_models, **risk_config)

    @staticmethod
    def _resolve_ensemble_weights(weight_cfg: Any, n_models: int) -> Optional[Sequence[float]]:
        if weight_cfg is None:
            return None
        if isinstance(weight_cfg, dict):
            ordered = [weight_cfg.get("lightgbm", 1.0), weight_cfg.get("xgboost", 1.0), weight_cfg.get("catboost", 1.0)]
            return ordered[:n_models]
        return weight_cfg
        
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """训练模型"""
        
        self.feature_columns = X.columns.tolist()
        self.model = self._build_model()

        fit_kwargs = {**self.fit_params, **kwargs}
        fit_X = X
        fit_y = y

        # 对于集成模型，跳过早停策略
        if (self.enable_early_stopping
            and self.model_type in {"lightgbm", "xgboost", "catboost"}
            and "eval_set" not in fit_kwargs
            and len(X) >= self.min_early_stopping_rows
            and 0 < self.auto_validation_fraction < 0.5
            and not self.model_type.endswith("ensemble")):
            
            val_size = max(32, int(len(X) * self.auto_validation_fraction))
            train_size = len(X) - val_size
            if train_size > 0:
                fit_X = X.iloc[:train_size]
                fit_y = y.iloc[:train_size]
                X_val = X.iloc[train_size:]
                y_val = y.iloc[train_size:]
                if self.model_type == "catboost":
                    fit_kwargs.setdefault("eval_set", (X_val, y_val))
                else:
                    fit_kwargs.setdefault("eval_set", [(X_val, y_val)])
                fit_kwargs.setdefault("early_stopping_rounds", 200)

        if self.model_type == "lightgbm" and "early_stopping_rounds" in fit_kwargs:
            early_rounds = fit_kwargs.pop("early_stopping_rounds")
            callbacks = list(fit_kwargs.get("callbacks") or [])
            try:
                from lightgbm import early_stopping as lgb_early_stopping
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError("LightGBM 未安装") from exc

            callbacks.append(lgb_early_stopping(stopping_rounds=early_rounds, verbose=False))
            fit_kwargs["callbacks"] = callbacks

        self.model.fit(fit_X, fit_y, **fit_kwargs)
        
    def predict(self, X: pd.DataFrame, *, clip: bool = True, noise_std: float = 0.0, 
               return_uncertainty: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """预测

        Args:
            X: 特征矩阵
            clip: 是否将预测值裁剪到[0,2]区间
            noise_std: 噪声标准差，用于增加预测变异性
            return_uncertainty: 是否返回不确定性估计
        """
        
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 检查模型是否支持不确定性返回
        supports_uncertainty = hasattr(self.model, 'predict') and 'return_uncertainty' in str(inspect.signature(self.model.predict))
        
        if return_uncertainty and supports_uncertainty:
            predictions, uncertainty = self.model.predict(X, return_uncertainty=True)
        else:
            predictions = self.model.predict(X)
            uncertainty = None
        
        # 添加噪声以增加预测变异性
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, size=predictions.shape)
            predictions = predictions + noise
        
        if clip:
            predictions = np.clip(predictions, 0, 2)
        
        if return_uncertainty and uncertainty is not None:
            return predictions, uncertainty
        else:
            return predictions
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict[str, float]:
        """时间序列交叉验证 - 增强版支持集成模型"""
        
        if not isinstance(y, (pd.Series, pd.DataFrame)):
            y = pd.Series(y, index=X.index)
        elif isinstance(y, pd.Series) and not y.index.equals(X.index):
            y = y.reindex(X.index)

        tscv = TimeSeriesSplit(n_splits=n_splits)
        mse_scores = []
        mae_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # 重新初始化模型以避免数据泄漏
            temp_model = HullModel(
                self.model_type, 
                self.model_params,
                ensemble_config=self.ensemble_config
            )
            temp_model.fit(X_train, y_train)
            
            y_pred = temp_model.predict(X_val, clip=False)
            
            mse_scores.append(mean_squared_error(y_val, y_pred))
            mae_scores.append(mean_absolute_error(y_val, y_pred))
        
        return {
            'mean_mse': np.mean(mse_scores),
            'std_mse': np.std(mse_scores),
            'mean_mae': np.mean(mae_scores),
            'std_mae': np.std(mae_scores)
        }
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """获取集成模型信息"""
        if not self.model_type.endswith("ensemble"):
            return {"type": "single_model"}
        
        info = {
            "type": self.model_type,
            "base_models": len(getattr(self.model, 'base_models', [])),
        }
        
        if hasattr(self.model, 'current_weights'):
            info["current_weights"] = self.model.current_weights.tolist()
        
        if hasattr(self.model, 'model_performances'):
            info["performance_monitored_models"] = list(self.model.model_performances.keys())
        
        return info


def create_submission(predictions: np.ndarray, date_ids: pd.Series) -> pd.DataFrame:
    """创建提交数据框"""
    
    submission_df = pd.DataFrame({
        'date_id': date_ids,
        'prediction': predictions
    })
    
    return submission_df


__all__ = [
    "HullModel",
    "create_baseline_model",
    "create_lightgbm_model",
    "create_xgboost_model",
    "create_catboost_model",
    "AveragingEnsemble",
    "DynamicWeightedEnsemble",
    "StackingEnsemble",
    "RiskAwareEnsemble",
    "AdversarialEnsemble",
    "MultiLevelEnsemble",
    "AdvancedModelPerformanceMonitor",
    "AdvancedConditionalWeightEngine",
    "RealTimePerformanceMonitor",
    "create_submission",
]
