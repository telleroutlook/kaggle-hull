"""
模型定义和训练工具
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error


def create_baseline_model():
    """创建基线模型"""
    
    # 导入配置
    try:
        from .config import get_config
        config = get_config()
        model_config = config.get_model_config()
        n_estimators = model_config['baseline_n_estimators']
        max_depth = model_config['baseline_max_depth']
        random_state = model_config['baseline_random_state']
    except ImportError:
        # 如果配置模块不可用，使用默认值
        n_estimators = 100
        max_depth = 10
        random_state = 42
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
    except ImportError:
        # 如果scikit-learn不可用，返回简单模型
        class SimpleModel:
            def __init__(self):
                self.coef_ = None
                
            def fit(self, X, y):
                # 简单的线性回归
                X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
                self.coef_ = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
                
            def predict(self, X):
                X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
                return X_with_intercept @ self.coef_
        
        return SimpleModel()


class HullModel:
    """Hull Tactical 预测模型"""
    
    def __init__(self, model_type: str = "baseline"):
        self.model_type = model_type
        self.model = None
        self.feature_columns = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """训练模型"""
        
        self.feature_columns = X.columns.tolist()
        
        if self.model_type == "baseline":
            self.model = create_baseline_model()
        else:
            # 可以在这里添加其他模型类型
            self.model = create_baseline_model()
        
        self.model.fit(X, y, **kwargs)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        predictions = self.model.predict(X)
        
        # 确保预测值在0-2之间
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
            temp_model = HullModel(self.model_type)
            temp_model.fit(X_train, y_train)
            
            y_pred = temp_model.predict(X_val)
            
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
    "create_submission",
]
