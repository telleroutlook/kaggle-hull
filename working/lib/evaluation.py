"""
评估和指标计算工具
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """计算夏普比率"""
    
    # Convert to numpy array and handle inf/nan values
    returns = np.asarray(returns, dtype=float)
    returns = np.where(np.isfinite(returns), returns, 0.0)
    
    excess_returns = returns - risk_free_rate
    
    if len(excess_returns) == 0:
        return 0.0
    
    # Check for valid standard deviation
    std_dev = np.std(excess_returns)
    if not np.isfinite(std_dev) or std_dev == 0:
        return 0.0
    
    mean_return = np.mean(excess_returns)
    if not np.isfinite(mean_return):
        return 0.0
    
    return mean_return / std_dev


def calculate_volatility_adjusted_sharpe(returns: np.ndarray, 
                                       market_volatility: float,
                                       risk_free_rate: float = 0.0,
                                       volatility_constraint: float = 1.2) -> float:
    """
    计算波动率调整的夏普比率
    
    Args:
        returns: 策略收益
        market_volatility: 市场波动率
        risk_free_rate: 无风险利率
        volatility_constraint: 波动率约束（默认120%）
    """
    
    portfolio_volatility = np.std(returns)
    
    # 计算基础夏普比率
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate)
    
    # 应用波动率惩罚
    if portfolio_volatility > market_volatility * volatility_constraint:
        # 如果超过波动率约束，严重惩罚
        penalty = (portfolio_volatility / (market_volatility * volatility_constraint)) - 1
        sharpe *= max(0, 1 - penalty)
    
    return sharpe


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, 
                  risk_free_rate: float = 0.0) -> Dict[str, float]:
    """评估模型性能"""
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
    }
    
    # 计算收益相关指标
    returns = y_pred  # 这里假设预测值就是分配比例
    market_returns = y_true  # 这里假设真实值就是市场收益
    
    metrics['sharpe_ratio'] = calculate_sharpe_ratio(returns, risk_free_rate)
    metrics['market_sharpe'] = calculate_sharpe_ratio(market_returns, risk_free_rate)
    
    return metrics


def backtest_strategy(allocation: np.ndarray, market_returns: np.ndarray,
                     risk_free_rate: float = 0.0) -> Dict[str, float]:
    """回测策略性能"""
    
    # Convert to numpy arrays and clean data
    allocation = np.asarray(allocation, dtype=float)
    market_returns = np.asarray(market_returns, dtype=float)
    
    # Replace inf/nan values with safe defaults
    allocation = np.where(np.isfinite(allocation), allocation, 1.0)
    market_returns = np.where(np.isfinite(market_returns), market_returns, 0.0)
    
    # 计算策略收益
    strategy_returns = allocation * market_returns
    
    # 计算指标，添加错误处理
    try:
        # Safe computation of total returns
        strategy_returns_safe = np.where(np.isfinite(strategy_returns), strategy_returns, 0.0)
        market_returns_safe = np.where(np.isfinite(market_returns), market_returns, 0.0)
        
        total_return = np.prod(1 + strategy_returns_safe) - 1
        market_total_return = np.prod(1 + market_returns_safe) - 1
        
        # Handle potential overflow in prod calculation
        if not np.isfinite(total_return):
            total_return = 0.0
        if not np.isfinite(market_total_return):
            market_total_return = 0.0
            
        metrics = {
            'strategy_total_return': total_return,
            'market_total_return': market_total_return,
            'excess_return': total_return - market_total_return,
            'strategy_sharpe': calculate_sharpe_ratio(strategy_returns, risk_free_rate),
            'market_sharpe': calculate_sharpe_ratio(market_returns, risk_free_rate),
            'strategy_volatility': np.std(strategy_returns) if len(strategy_returns) > 0 else 0.0,
            'market_volatility': np.std(market_returns) if len(market_returns) > 0 else 0.0,
            'max_allocation': np.max(allocation) if len(allocation) > 0 else 0.0,
            'min_allocation': np.min(allocation) if len(allocation) > 0 else 0.0,
            'mean_allocation': np.mean(allocation) if len(allocation) > 0 else 0.0,
        }
    except (OverflowError, ValueError, ZeroDivisionError):
        # Fallback metrics in case of calculation errors
        metrics = {
            'strategy_total_return': 0.0,
            'market_total_return': 0.0,
            'excess_return': 0.0,
            'strategy_sharpe': 0.0,
            'market_sharpe': 0.0,
            'strategy_volatility': 0.0,
            'market_volatility': 0.0,
            'max_allocation': 1.0,
            'min_allocation': 1.0,
            'mean_allocation': 1.0,
        }
    
    return metrics


__all__ = [
    "calculate_sharpe_ratio",
    "calculate_volatility_adjusted_sharpe", 
    "evaluate_model",
    "backtest_strategy",
]