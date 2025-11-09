"""
模型定义和训练工具
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Sequence
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error


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
    """Hull Tactical 预测模型"""
    
    def __init__(
        self,
        model_type: str = "baseline",
        model_params: Optional[Dict[str, Any]] = None,
        *,
        random_state: Optional[int] = 42,
        auto_validation_fraction: float = 0.1,
        enable_early_stopping: bool = True,
    ):
        self.model_type = model_type.lower()
        params = dict(model_params or {})
        self.fit_params = params.pop("fit_params", {})
        self.model_params = params
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
            ensemble_params = dict(self.model_params)
            weight_cfg = ensemble_params.pop("weights", None)
            base_models = [
                create_lightgbm_model(random_state=self.random_state, **ensemble_params.get("lightgbm", {})),
                create_xgboost_model(random_state=self.random_state, **ensemble_params.get("xgboost", {})),
                create_catboost_model(random_state=self.random_state, **ensemble_params.get("catboost", {})),
            ]
            weights = self._resolve_ensemble_weights(weight_cfg, len(base_models))
            return AveragingEnsemble(base_models, weights=weights)
        raise ValueError(f"Unknown model type: {self.model_type}")

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

        if (
            self.enable_early_stopping
            and self.model_type in {"lightgbm", "xgboost", "catboost"}
            and "eval_set" not in fit_kwargs
            and len(X) >= self.min_early_stopping_rows
            and 0 < self.auto_validation_fraction < 0.5
        ):
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
