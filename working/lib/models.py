"""
模型定义和训练工具
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Sequence
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error


def create_baseline_model(**overrides):
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

    default_params.update(overrides)

    try:
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=default_params.get("n_estimators", 200),
            max_depth=default_params.get("max_depth", 12),
            random_state=default_params.get("random_state", 42),
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


def create_lightgbm_model(**overrides):
    """LightGBM 模型"""

    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise ImportError("LightGBM 未安装") from exc

    params = {
        "objective": "regression",
        "n_estimators": 4000,
        "learning_rate": 0.005,
        "num_leaves": 256,
        "max_depth": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 5.0,
        "min_child_samples": 40,
        "n_jobs": -1,
        "verbosity": -1,
    }
    params.update(overrides)
    return LGBMRegressor(**params)


def create_xgboost_model(**overrides):
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
    params.update(overrides)
    return XGBRegressor(**params)


def create_catboost_model(**overrides):
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
    params.update(overrides)
    return CatBoostRegressor(**params)


class AveragingEnsemble:
    """简单均值集成器"""

    def __init__(self, base_models: Sequence):
        self.base_models = base_models

    def fit(self, X, y):
        for model in self.base_models:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.vstack([model.predict(X) for model in self.base_models])
        return predictions.mean(axis=0)


class HullModel:
    """Hull Tactical 预测模型"""
    
    def __init__(self, model_type: str = "baseline", model_params: Optional[Dict[str, Any]] = None):
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model = None
        self.feature_columns = None

    def _build_model(self):
        if self.model_type == "baseline":
            return create_baseline_model(**self.model_params)
        if self.model_type == "lightgbm":
            return create_lightgbm_model(**self.model_params)
        if self.model_type == "xgboost":
            return create_xgboost_model(**self.model_params)
        if self.model_type == "catboost":
            return create_catboost_model(**self.model_params)
        if self.model_type == "ensemble":
            base_models = [
                create_lightgbm_model(**self.model_params.get("lightgbm", {})),
                create_xgboost_model(**self.model_params.get("xgboost", {})),
                create_catboost_model(**self.model_params.get("catboost", {})),
            ]
            return AveragingEnsemble(base_models)
        raise ValueError(f"Unknown model type: {self.model_type}")
        
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """训练模型"""
        
        self.feature_columns = X.columns.tolist()
        
        self.model = self._build_model()
        self.model.fit(X, y, **kwargs)
        
    def predict(self, X: pd.DataFrame, *, clip: bool = True) -> np.ndarray:
        """预测

        Args:
            X: 特征矩阵
            clip: 是否将预测值裁剪到[0,2]区间。训练/策略阶段通常需要原始值。
        """
        
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        predictions = self.model.predict(X)
        
        if clip:
            predictions = np.clip(predictions, 0, 2)
        
        return predictions
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict[str, float]:
        """时间序列交叉验证"""
        
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
            temp_model = HullModel(self.model_type, self.model_params)
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
    "create_submission",
]
