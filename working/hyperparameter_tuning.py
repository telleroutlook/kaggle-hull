#!/usr/bin/env python3
"""
Hull Tactical - 智能超参数调优系统
支持多模型、贝叶斯优化、时间序列验证和多目标优化
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
from datetime import datetime
import copy
import threading
from collections import defaultdict, deque
import pickle

# 数据科学和ML库
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import optuna
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.pruners.successive_halving import SuccessiveHalvingPruner

# 项目相关
from lib.data import load_train_data
from lib.features import FeaturePipeline
from lib.models import (
    create_lightgbm_model, 
    create_xgboost_model, 
    create_catboost_model,
    HullModel
)
from lib.utils import PerformanceTracker

warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TuningConfig:
    """调优配置"""
    # 基础配置
    model_types: List[str]  # 要调优的模型类型
    n_trials: int = 100  # Optuna试验次数
    cv_folds: int = 5  # 交叉验证折叠数
    random_state: int = 42
    
    # 搜索策略
    search_strategy: str = "optuna"  # grid, random, optuna, mixed
    early_stopping_rounds: int = 50
    timeout_seconds: int = 3600  # 1小时超时
    
    # 验证策略
    validation_strategy: str = "time_series"  # time_series, rolling, expanding
    test_size: float = 0.2
    
    # 多目标优化
    primary_metric: str = "mse"  # mse, mae, r2
    secondary_metrics: List[str] = None  # 多目标指标
    
    # 性能约束
    max_training_time: float = 300  # 单次训练最大时间(秒)
    min_improvement: float = 0.001  # 最小改进阈值
    patience: int = 10  # 早停耐心
    
    # 输出配置
    save_results: bool = True
    output_dir: str = "tuning_results"
    save_intermediate: bool = True
    
    def __post_init__(self):
        if self.secondary_metrics is None:
            self.secondary_metrics = ["mae", "r2"]


@dataclass
class TuningResult:
    """调优结果"""
    model_type: str
    best_params: Dict[str, Any]
    best_score: float
    cv_scores: Dict[str, List[float]]
    tuning_time: float
    n_trials: int
    feature_importance: Optional[Dict[str, float]] = None
    learning_curve: Optional[List[Tuple[int, float]]] = None
    validation_history: Optional[List[Dict[str, float]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TimeSeriesValidator:
    """时间序列验证器"""
    
    def __init__(self, strategy: str = "time_series", n_splits: int = 5, 
                 test_size: float = 0.2, random_state: int = 42):
        self.strategy = strategy
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state
        
    def split(self, X: pd.DataFrame, y: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
        """生成训练/验证分割"""
        if self.strategy == "time_series":
            return self._time_series_split(X, y)
        elif self.strategy == "rolling":
            return self._rolling_window_split(X, y)
        elif self.strategy == "expanding":
            return self._expanding_window_split(X, y)
        else:
            raise ValueError(f"未知验证策略: {self.strategy}")
    
    def _time_series_split(self, X: pd.DataFrame, y: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
        """时间序列分割"""
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        splits = list(tscv.split(X))
        
        # 转换为索引格式
        train_val_splits = []
        for train_idx, val_idx in splits:
            train_val_splits.append((train_idx, val_idx))
        
        return train_val_splits
    
    def _rolling_window_split(self, X: pd.DataFrame, y: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
        """滚动窗口分割"""
        n_samples = len(X)
        test_window = int(n_samples * self.test_size)
        min_train_size = int(n_samples * 0.3)  # 最小训练集大小
        
        splits = []
        for i in range(self.n_splits):
            end_idx = n_samples - (i * test_window // self.n_splits)
            start_idx = max(min_train_size, end_idx - int(n_samples * 0.8))
            
            train_idx = np.arange(start_idx, end_idx - test_window)
            val_idx = np.arange(end_idx - test_window, end_idx)
            
            if len(train_idx) > 0 and len(val_idx) > 0:
                splits.append((train_idx, val_idx))
        
        return splits
    
    def _expanding_window_split(self, X: pd.DataFrame, y: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
        """扩展窗口分割"""
        n_samples = len(X)
        min_train_size = int(n_samples * 0.2)
        test_size = int(n_samples * self.test_size)
        
        splits = []
        for i in range(self.n_splits):
            end_train = min_train_size + (n_samples - min_train_size - test_size) * i // (self.n_splits - 1)
            start_test = end_train
            end_test = min(start_test + test_size, n_samples)
            
            train_idx = np.arange(0, start_test)
            val_idx = np.arange(start_test, end_test)
            
            if len(train_idx) > 0 and len(val_idx) > 0:
                splits.append((train_idx, val_idx))
        
        return splits


class MultiObjectiveEvaluator:
    """多目标评估器"""
    
    def __init__(self, primary_metric: str = "mse", secondary_metrics: List[str] = None):
        self.primary_metric = primary_metric
        self.secondary_metrics = secondary_metrics or ["mae", "r2"]
        
        self.metric_functions = {
            "mse": mean_squared_error,
            "mae": mean_absolute_error,
            "r2": r2_score
        }
        
        # 指标权重（可调整）
        self.secondary_weights = {
            "mae": 0.3,
            "r2": 0.2
        }
        
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """计算多个评估指标"""
        results = {}
        
        for metric in [self.primary_metric] + self.secondary_metrics:
            if metric in self.metric_functions:
                try:
                    if metric == "r2":
                        score = self.metric_functions[metric](y_true, y_pred)
                    else:
                        score = self.metric_functions[metric](y_true, y_pred)
                    results[metric] = score
                except Exception as e:
                    logger.warning(f"计算{metric}失败: {e}")
                    results[metric] = np.inf if metric == "mse" else 0.0
        
        return results
    
    def combine_scores(self, scores: Dict[str, float]) -> float:
        """组合多目标分数"""
        primary_score = scores.get(self.primary_metric, np.inf)
        
        if self.primary_metric == "mse" or self.primary_metric == "mae":
            # 对于误差指标，越小越好
            combined_score = primary_score
            for metric, weight in self.secondary_weights.items():
                if metric in scores and not np.isinf(scores[metric]):
                    # 将所有指标转换为最小化问题
                    if metric == "r2":
                        adjusted_score = 1 - max(0, scores[metric])  # 转换为误差
                    else:
                        adjusted_score = scores[metric]
                    combined_score += weight * adjusted_score
        else:
            # 对于其他指标（如r2），越大越好
            combined_score = 1.0 / (1.0 + primary_score)  # 转换为最小化
            for metric, weight in self.secondary_weights.items():
                if metric in scores and not np.isinf(scores[metric]):
                    if metric == "r2":
                        adjusted_score = 1 - max(0, scores[metric])
                    else:
                        adjusted_score = scores[metric]
                    combined_score += weight * adjusted_score
        
        return combined_score


class ParameterSpace:
    """参数空间定义"""
    
    @staticmethod
    def get_lightgbm_space() -> Dict[str, Any]:
        """LightGBM参数空间"""
        return {
            "n_estimators": optuna.distributions.IntDistribution(500, 5000, log=True),
            "learning_rate": optuna.distributions.FloatDistribution(0.005, 0.1, log=True),
            "num_leaves": optuna.distributions.IntDistribution(16, 512, log=True),
            "max_depth": optuna.distributions.IntDistribution(3, 15),
            "subsample": optuna.distributions.FloatDistribution(0.6, 1.0),
            "colsample_bytree": optuna.distributions.FloatDistribution(0.6, 1.0),
            "reg_alpha": optuna.distributions.FloatDistribution(0, 10, log=True),
            "reg_lambda": optuna.distributions.FloatDistribution(0, 10, log=True),
            "min_child_samples": optuna.distributions.IntDistribution(5, 50),
            "min_child_weight": optuna.distributions.FloatDistribution(0.001, 10, log=True),
            "boosting_type": optuna.distributions.CategoricalDistribution(["gbdt", "dart", "goss"])
        }
    
    @staticmethod
    def get_xgboost_space() -> Dict[str, Any]:
        """XGBoost参数空间"""
        return {
            "n_estimators": optuna.distributions.IntDistribution(500, 5000, log=True),
            "learning_rate": optuna.distributions.FloatDistribution(0.005, 0.1, log=True),
            "max_depth": optuna.distributions.IntDistribution(3, 12),
            "subsample": optuna.distributions.FloatDistribution(0.6, 1.0),
            "colsample_bytree": optuna.distributions.FloatDistribution(0.6, 1.0),
            "colsample_bylevel": optuna.distributions.FloatDistribution(0.6, 1.0),
            "reg_alpha": optuna.distributions.FloatDistribution(0, 10, log=True),
            "reg_lambda": optuna.distributions.FloatDistribution(0, 10, log=True),
            "gamma": optuna.distributions.FloatDistribution(0, 5, log=True),
            "min_child_weight": optuna.distributions.FloatDistribution(1, 10, log=True),
            "scale_pos_weight": optuna.distributions.FloatDistribution(0.5, 2.0),
            "tree_method": optuna.distributions.CategoricalDistribution(["hist", "approx"])
        }
    
    @staticmethod
    def get_catboost_space() -> Dict[str, Any]:
        """CatBoost参数空间"""
        return {
            "iterations": optuna.distributions.IntDistribution(500, 5000, log=True),
            "learning_rate": optuna.distributions.FloatDistribution(0.005, 0.1, log=True),
            "depth": optuna.distributions.IntDistribution(4, 12),
            "l2_leaf_reg": optuna.distributions.FloatDistribution(1, 10, log=True),
            "border_count": optuna.distributions.IntDistribution(32, 255),
            "bagging_temperature": optuna.distributions.FloatDistribution(0, 1),
            "random_strength": optuna.distributions.FloatDistribution(0, 2),
            "min_data_in_leaf": optuna.distributions.IntDistribution(1, 50),
            "rsm": optuna.distributions.FloatDistribution(0.6, 1.0),  # similar to colsample_bytree
            "leaf_estimation_iterations": optuna.distributions.IntDistribution(1, 20),
            "boosting_type": optuna.distributions.CategoricalDistribution(["Ordered", "Plain"])
        }
    
    @staticmethod
    def get_random_forest_space() -> Dict[str, Any]:
        """随机森林参数空间"""
        return {
            "n_estimators": optuna.distributions.IntDistribution(50, 500, log=True),
            "max_depth": optuna.distributions.IntDistribution(5, 50),
            "min_samples_split": optuna.distributions.IntDistribution(2, 20),
            "min_samples_leaf": optuna.distributions.IntDistribution(1, 20),
            "max_features": optuna.distributions.CategoricalDistribution(["sqrt", "log2", 0.5, 0.8]),
            "bootstrap": optuna.distributions.CategoricalDistribution([True, False])
        }
    
    @staticmethod
    def get_parameter_space(model_type: str) -> Dict[str, Any]:
        """获取模型参数空间"""
        space_map = {
            "lightgbm": ParameterSpace.get_lightgbm_space(),
            "xgboost": ParameterSpace.get_xgboost_space(),
            "catboost": ParameterSpace.get_catboost_space(),
            "random_forest": ParameterSpace.get_random_forest_space()
        }
        
        if model_type.lower() not in space_map:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        return space_map[model_type.lower()]


class AdaptiveSearchStrategy:
    """自适应搜索策略"""
    
    def __init__(self, strategy: str = "optuna", random_state: int = 42):
        self.strategy = strategy
        self.random_state = random_state
        self.trial_history = []
        self.performance_history = deque(maxlen=50)
        self.search_phase = "exploration"  # exploration, exploitation, refinement
        
    def should_stop_early(self, trial: optuna.Trial) -> bool:
        """早停判断"""
        if len(self.performance_history) < 5:
            return False
        
        recent_scores = list(self.performance_history)[-10:]
        if len(recent_scores) < 3:
            return False
        
        # 检查性能是否停滞
        improvements = []
        for i in range(1, len(recent_scores)):
            if recent_scores[i] < recent_scores[i-1]:  # 假设最小化
                improvements.append(recent_scores[i-1] - recent_scores[i])
        
        if len(improvements) < 3:
            return False
        
        avg_improvement = np.mean(improvements)
        return avg_improvement < 0.001  # 改进太小
    
    def update_performance(self, score: float):
        """更新性能历史"""
        self.performance_history.append(score)
    
    def suggest_parameters(self, trial: optuna.Trial, param_space: Dict[str, Any]) -> Dict[str, Any]:
        """建议参数"""
        params = {}
        
        for param_name, distribution in param_space.items():
            try:
                if isinstance(distribution, optuna.distributions.CategoricalDistribution):
                    params[param_name] = trial.suggest_categorical(param_name, distribution.choices)
                elif isinstance(distribution, optuna.distributions.IntDistribution):
                    if distribution.log:
                        params[param_name] = trial.suggest_int(param_name, distribution.low, distribution.high, log=True)
                    else:
                        params[param_name] = trial.suggest_int(param_name, distribution.low, distribution.high)
                elif isinstance(distribution, optuna.distributions.FloatDistribution):
                    if distribution.log:
                        params[param_name] = trial.suggest_float(param_name, distribution.low, distribution.high, log=True)
                    else:
                        params[param_name] = trial.suggest_float(param_name, distribution.low, distribution.high)
            except Exception as e:
                logger.warning(f"参数建议失败 {param_name}: {e}")
                # 使用默认值
                if hasattr(distribution, 'low') and hasattr(distribution, 'high'):
                    if distribution.low <= 0 <= distribution.high:
                        params[param_name] = 0
                    else:
                        params[param_name] = distribution.low
        
        return params


class HyperparameterTuner:
    """主超参数调优器"""
    
    def __init__(self, config: TuningConfig):
        self.config = config
        self.validator = TimeSeriesValidator(
            strategy=config.validation_strategy,
            n_splits=config.cv_folds,
            test_size=config.test_size,
            random_state=config.random_state
        )
        self.evaluator = MultiObjectiveEvaluator(
            primary_metric=config.primary_metric,
            secondary_metrics=config.secondary_metrics
        )
        self.search_strategy = AdaptiveSearchStrategy(
            strategy=config.search_strategy,
            random_state=config.random_state
        )
        self.results = {}
        self.best_models = {}
        
        # 创建输出目录
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 设置随机种子
        np.random.seed(config.random_state)
        
    def _create_model(self, model_type: str, params: Dict[str, Any]) -> Any:
        """创建模型实例"""
        if model_type.lower() == "lightgbm":
            return create_lightgbm_model(random_state=self.config.random_state, **params)
        elif model_type.lower() == "xgboost":
            return create_xgboost_model(random_state=self.config.random_state, **params)
        elif model_type.lower() == "catboost":
            return create_catboost_model(random_state=self.config.random_state, **params)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def _evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """评估模型"""
        start_time = time.time()
        cv_scores = defaultdict(list)
        validation_history = []
        
        splits = self.validator.split(X, y)
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # 训练模型
            try:
                # 创建新模型实例以避免数据泄漏
                temp_model = copy.deepcopy(model)
                temp_model.fit(X_train, y_train)
                
                # 预测
                y_pred = temp_model.predict(X_val)
                
                # 计算指标
                fold_scores = self.evaluator.evaluate(y_val.values, y_pred)
                for metric, score in fold_scores.items():
                    cv_scores[metric].append(score)
                
                # 记录验证历史
                validation_history.append({
                    "fold": fold,
                    "train_size": len(X_train),
                    "val_size": len(X_val),
                    "scores": fold_scores
                })
                
            except Exception as e:
                logger.warning(f"第{fold}折评估失败: {e}")
                # 为失败的折添加默认分数
                for metric in [self.config.primary_metric] + self.config.secondary_metrics:
                    cv_scores[metric].append(np.inf if metric in ["mse", "mae"] else 0.0)
        
        # 计算平均分数
        mean_scores = {metric: np.mean(scores) for metric, scores in cv_scores.items()}
        std_scores = {metric: np.std(scores) for metric, scores in cv_scores.items()}
        
        # 组合分数
        combined_score = self.evaluator.combine_scores(mean_scores)
        
        evaluation_time = time.time() - start_time
        
        return {
            "cv_scores": dict(cv_scores),
            "mean_scores": mean_scores,
            "std_scores": std_scores,
            "combined_score": combined_score,
            "evaluation_time": evaluation_time,
            "validation_history": validation_history
        }
    
    def _objective(self, trial: optuna.Trial, model_type: str, X: pd.DataFrame, y: pd.Series) -> float:
        """Optuna目标函数"""
        # 获取参数空间
        param_space = ParameterSpace.get_parameter_space(model_type)
        
        # 建议参数
        params = self.search_strategy.suggest_parameters(trial, param_space)
        
        try:
            # 创建模型
            model = self._create_model(model_type, params)
            
            # 评估模型
            results = self._evaluate_model(model, X, y)
            
            # 更新搜索策略
            self.search_strategy.update_performance(results["combined_score"])
            
            # 早停检查
            if self.search_strategy.should_stop_early(trial):
                raise optuna.TrialPruned()
            
            return results["combined_score"]
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.warning(f"试验失败: {e}")
            # 返回差的分数
            if self.config.primary_metric in ["mse", "mae"]:
                return 1e6
            else:
                return -1e6
    
    def tune_single_model(self, model_type: str, X: pd.DataFrame, y: pd.Series) -> TuningResult:
        """调优单个模型"""
        logger.info(f"开始调优 {model_type} 模型...")
        start_time = time.time()
        
        # 创建Optuna研究
        study_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            direction="minimize" if self.config.primary_metric in ["mse", "mae"] else "maximize",
            study_name=study_name,
            sampler=TPESampler(seed=self.config.random_state),
            pruner=MedianPruner(n_warmup_steps=5)
        )
        
        # 设置超时
        timeout = min(self.config.timeout_seconds, 3600)  # 最大1小时
        
        try:
            # 优化
            study.optimize(
                lambda trial: self._objective(trial, model_type, X, y),
                n_trials=self.config.n_trials,
                timeout=timeout,
                n_jobs=1  # 单线程避免内存问题
            )
            
        except Exception as e:
            logger.error(f"调优失败: {e}")
            # 如果调优失败，使用默认参数
            default_params = {}
            best_params = default_params
            best_score = float('inf')
        else:
            # 获取最佳参数和分数
            best_params = study.best_params
            best_score = study.best_value
        
        tuning_time = time.time() - start_time
        
        # 使用最佳参数进行最终评估
        final_model = self._create_model(model_type, best_params)
        final_results = self._evaluate_model(final_model, X, y)
        
        # 创建调优结果
        result = TuningResult(
            model_type=model_type,
            best_params=best_params,
            best_score=best_score,
            cv_scores=final_results["cv_scores"],
            tuning_time=tuning_time,
            n_trials=len(study.trials)
        )
        
        logger.info(f"{model_type} 调优完成 - 最佳分数: {best_score:.6f}, 耗时: {tuning_time:.2f}秒")
        
        return result
    
    def tune_all_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, TuningResult]:
        """调优所有模型"""
        logger.info(f"开始调优 {len(self.config.model_types)} 个模型...")
        
        for model_type in self.config.model_types:
            try:
                result = self.tune_single_model(model_type, X, y)
                self.results[model_type] = result
                self.best_models[model_type] = self._create_model(model_type, result.best_params)
            except Exception as e:
                logger.error(f"调优 {model_type} 失败: {e}")
                continue
        
        return self.results
    
    def get_ranking(self) -> List[Tuple[str, float]]:
        """获取模型排名"""
        rankings = []
        for model_type, result in self.results.items():
            rankings.append((model_type, result.best_score))
        
        # 排序
        if self.config.primary_metric in ["mse", "mae"]:
            rankings.sort(key=lambda x: x[1])  # 升序
        else:
            rankings.sort(key=lambda x: x[1], reverse=True)  # 降序
        
        return rankings
    
    def save_results(self, filename: str = None):
        """保存调优结果"""
        if filename is None:
            filename = f"tuning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # 准备保存数据
        save_data = {
            "config": asdict(self.config),
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "rankings": self.get_ranking(),
            "timestamp": datetime.now().isoformat()
        }
        
        # 保存为JSON
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"调优结果已保存到: {output_path}")
        
        # 同时保存为pickle格式（包含更多详细信息）
        pickle_path = self.output_dir / f"{filename.replace('.json', '.pkl')}"
        with open(pickle_path, 'wb') as f:
            pickle.dump({
                "results": self.results,
                "best_models": self.best_models,
                "config": self.config
            }, f)
        
        logger.info(f"详细结果已保存到: {pickle_path}")


def run_tuning_experiments(data_path: str = None, output_dir: str = "tuning_results"):
    """运行完整的调优实验"""
    
    # 默认配置
    config = TuningConfig(
        model_types=["lightgbm", "xgboost", "catboost"],
        n_trials=50,  # 减少试验次数以节省时间
        cv_folds=3,
        validation_strategy="time_series",
        search_strategy="optuna",
        timeout_seconds=1800,  # 30分钟超时
        primary_metric="mse",
        secondary_metrics=["mae", "r2"],
        output_dir=output_dir
    )
    
    logger.info("开始超参数调优实验...")
    
    # 加载数据
    if data_path is None:
        data_path = "input/hull-tactical-market-prediction"
    
    try:
        # 加载训练数据
        train_data_path = Path(data_path) / "train.csv"
        if not train_data_path.exists():
            raise FileNotFoundError(f"训练数据文件不存在: {train_data_path}")
        
        train_data = pd.read_csv(train_data_path)
        logger.info(f"加载训练数据: {train_data.shape}")
        
        # 特征工程
        logger.info("开始特征工程...")
        pipeline = FeaturePipeline(stateful=True)
        X = pipeline.fit_transform(train_data)
        y = train_data["forward_returns"].fillna(train_data["forward_returns"].median())
        
        logger.info(f"特征工程完成: {X.shape}")
        
        # 初始化调优器
        tuner = HyperparameterTuner(config)
        
        # 执行调优
        results = tuner.tune_all_models(X, y)
        
        # 显示结果
        logger.info("=" * 50)
        logger.info("调优结果总结:")
        logger.info("=" * 50)
        
        rankings = tuner.get_ranking()
        for i, (model_type, score) in enumerate(rankings, 1):
            result = results[model_type]
            logger.info(f"{i}. {model_type}: {score:.6f} (耗时: {result.tuning_time:.2f}s, 试验数: {result.n_trials})")
        
        # 保存结果
        tuner.save_results()
        
        logger.info("超参数调优实验完成！")
        return tuner, results
        
    except Exception as e:
        logger.error(f"调优实验失败: {e}")
        raise


if __name__ == "__main__":
    # 运行调优实验
    tuner, results = run_tuning_experiments()
    print(f"\n最佳模型: {tuner.get_ranking()[0][0]}")
    print(f"调优结果保存在: {tuner.output_dir}")
