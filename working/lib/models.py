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


class ModelPerformanceMonitor:
    """模型性能监控器"""
    
    def __init__(self, window_size: int = 100, update_frequency: int = 10):
        self.window_size = window_size
        self.update_frequency = update_frequency
        self.model_performances = defaultdict(lambda: deque(maxlen=window_size))
        self.last_update_idx = 0
        self.lock = threading.RLock()
        
    def update_performance(self, model_name: str, true_values: np.ndarray, 
                          predictions: np.ndarray, update_idx: int):
        """更新模型性能指标"""
        with self.lock:
            if update_idx - self.last_update_idx >= self.update_frequency or self.last_update_idx == 0:
                # 计算MSE和相关性
                mse = mean_squared_error(true_values, predictions)
                correlation = np.corrcoef(true_values, predictions)[0, 1] if len(true_values) > 1 else 0.0
                
                # 计算移动平均和波动性指标
                if len(self.model_performances[model_name]) > 0:
                    recent_mse = np.mean([p['mse'] for p in self.model_performances[model_name]])
                    recent_corr = np.mean([p['correlation'] for p in self.model_performances[model_name]])
                    
                    # 计算MSE趋势和稳定性
                    mse_trend = (mse - recent_mse) / (recent_mse + 1e-8)
                    stability_score = 1.0 / (1.0 + abs(mse_trend))
                else:
                    stability_score = 1.0
                
                performance_score = {
                    'mse': mse,
                    'correlation': correlation if not np.isnan(correlation) else 0.0,
                    'stability_score': stability_score,
                    'timestamp': update_idx,
                    'sample_size': len(true_values)
                }
                
                self.model_performances[model_name].append(performance_score)
                self.last_update_idx = update_idx
                
    def get_model_weights(self, model_names: List[str], 
                         performance_boost: float = 0.1,
                         stability_boost: float = 0.1) -> Dict[str, float]:
        """基于历史性能计算动态权重"""
        with self.lock:
            weights = {}
            total_weight = 0.0
            
            for model_name in model_names:
                if model_name not in self.model_performances:
                    # 新模型，默认权重
                    weights[model_name] = 1.0 / len(model_names)
                    continue
                
                perf_history = list(self.model_performances[model_name])
                if not perf_history:
                    weights[model_name] = 1.0 / len(model_names)
                    continue
                
                # 计算综合性能指标
                recent_perf = perf_history[-min(5, len(perf_history)):]  # 最近5个记录
                avg_mse = np.mean([p['mse'] for p in recent_perf])
                avg_corr = np.mean([p['correlation'] for p in recent_perf])
                avg_stability = np.mean([p['stability_score'] for p in recent_perf])
                
                # 性能权重计算 (低MSE + 高相关性 + 高稳定性)
                base_score = (1.0 / (1.0 + avg_mse))  # MSE倒数
                correlation_score = max(0, avg_corr)  # 相关性(负值惩罚)
                combined_score = (base_score * (1.0 + performance_boost) + 
                                correlation_score * (1.0 + stability_boost) +
                                avg_stability * stability_boost)
                
                weights[model_name] = max(combined_score, 1e-8)  # 避免零权重
                total_weight += weights[model_name]
            
            # 归一化权重
            if total_weight > 0:
                for model_name in weights:
                    weights[model_name] /= total_weight
            else:
                # 均匀分配
                equal_weight = 1.0 / len(model_names)
                weights = {name: equal_weight for name in model_names}
                
            return weights


class ConditionalWeightEngine:
    """条件化权重引擎，基于市场状态动态调整权重"""
    
    def __init__(self, market_lookback: int = 60):
        self.market_lookback = market_lookback
        self.market_states = deque(maxlen=market_lookback)
        self.state_weights = {
            'trending_up': {'lightgbm': 0.4, 'xgboost': 0.4, 'catboost': 0.2},
            'trending_down': {'lightgbm': 0.3, 'xgboost': 0.3, 'catboost': 0.4},
            'volatile': {'lightgbm': 0.2, 'xgboost': 0.3, 'catboost': 0.5},
            'stable': {'lightgbm': 0.5, 'xgboost': 0.3, 'catboost': 0.2},
            'unknown': {'lightgbm': 1/3, 'xgboost': 1/3, 'catboost': 1/3}
        }
        
    def update_market_state(self, market_returns: np.ndarray, volatility_threshold: float = 0.015):
        """更新市场状态"""
        if len(market_returns) < 2:
            return 'unknown'
            
        # 计算趋势和波动性
        recent_returns = market_returns[-min(20, len(market_returns)):]
        trend = np.mean(recent_returns)  # 平均收益率作为趋势指标
        volatility = np.std(recent_returns)  # 波动性
        
        # 确定市场状态
        if volatility > volatility_threshold * 2:
            state = 'volatile'
        elif trend > 0.01:  # 明显上涨趋势
            state = 'trending_up'
        elif trend < -0.01:  # 明显下跌趋势
            state = 'trending_down'
        else:
            state = 'stable'
            
        self.market_states.append(state)
        return state
    
    def get_conditional_weights(self, current_state: str) -> Dict[str, float]:
        """获取当前市场状态下的条件化权重"""
        return self.state_weights.get(current_state, self.state_weights['unknown'])


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
        self.performance_monitor = ModelPerformanceMonitor(window_size=performance_window)
        self.conditional_engine = ConditionalWeightEngine()
        self.current_weights = self.initial_weights.copy()
        self.prediction_count = 0
        self.model_failures = defaultdict(int)
        self.last_successful_predictions = defaultdict(int)
        
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
    
    def _update_weights(self, predictions: np.ndarray, true_values: Optional[np.ndarray] = None):
        """更新模型权重"""
        if true_values is None:
            # 无真实值时使用平滑回退
            fallback_weights = self._get_fallback_weights()
            smoothing_factor = self.weight_smoothing
        else:
            # 基于性能更新权重
            with ThreadPoolExecutor(max_workers=1) as executor:
                futures = []
                for i, (model, model_name) in enumerate(zip(self.base_models, self.model_names)):
                    future = executor.submit(
                        self.performance_monitor.update_performance,
                        model_name, true_values, predictions[:, i] if predictions.ndim > 1 else predictions,
                        self.prediction_count
                    )
                    futures.append(future)
                
                # 等待所有更新完成
                for future in futures:
                    try:
                        future.result(timeout=5.0)  # 5秒超时
                    except Exception as e:
                        print(f"⚠️ 性能更新失败: {e}")
            
            # 获取性能权重
            perf_weights_dict = self.performance_monitor.get_model_weights(self.model_names)
            perf_weights = np.array([perf_weights_dict[name] for name in self.model_names])
            
            # 如果启用条件化权重，混合市场状态权重
            if self.conditional_weighting and len(true_values) > 0:
                # 假设市场状态基于真实值计算（这里需要市场数据，实际实现时需要调整）
                market_state = self.conditional_engine.update_market_state(true_values)
                conditional_weights_dict = self.conditional_engine.get_conditional_weights(market_state)
                conditional_weights = np.array([conditional_weights_dict.get(name.split('_')[1], 1/3) 
                                              for name in self.model_names])
                
                # 混合权重：70%性能权重 + 30%条件化权重
                mixed_weights = 0.7 * perf_weights + 0.3 * conditional_weights
                fallback_weights = mixed_weights
            else:
                fallback_weights = perf_weights
            
            smoothing_factor = self.weight_smoothing
        
        # 应用权重平滑
        self.current_weights = (1 - smoothing_factor) * self.current_weights + smoothing_factor * fallback_weights
        
        # 应用最小权重阈值
        min_total = self.min_weight_threshold * len(self.base_models)
        if self.current_weights.sum() < min_total:
            self.current_weights = np.ones(len(self.base_models)) / len(self.base_models)
        
        # 确保权重和为1
        self.current_weights /= self.current_weights.sum()
    
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
        
        # 更新权重（如果有真实值，这里需要调整以支持在线学习）
        if self.prediction_count % 10 == 0:  # 每10次预测更新一次权重
            self._update_weights(all_predictions, None)  # 暂时不使用真实值更新
        
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


class RiskAwareEnsemble:
    """风险感知集成器，结合预测不确定性和风险约束"""
    
    def __init__(self, base_models: Sequence, risk_models: Optional[Sequence] = None,
                 volatility_constraint: float = 1.2, uncertainty_threshold: float = 0.1,
                 risk_parity: bool = True):
        self.base_models = list(base_models)
        self.risk_models = risk_models or base_models
        self.volatility_constraint = volatility_constraint
        self.uncertainty_threshold = uncertainty_threshold
        self.risk_parity = risk_parity
        
        self.model_uncertainties = defaultdict(list)
        self.risk_adjusted_weights = None
        
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
                                   uncertainties: np.ndarray) -> np.ndarray:
        """计算风险贡献"""
        # 简化的风险贡献计算
        if uncertainties is None:
            uncertainties = np.ones_like(predictions) * 0.1
            
        # 基于不确定性的风险调整
        risk_factor = 1.0 / (1.0 + uncertainties)
        return risk_factor
    
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
    "ModelPerformanceMonitor",
    "ConditionalWeightEngine",
    "create_submission",
]
