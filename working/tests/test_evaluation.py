"""
测试评估功能
"""

import sys
import os
import pytest
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from lib.evaluation import (
        calculate_sharpe_ratio,
        calculate_volatility_adjusted_sharpe,
        evaluate_model,
        backtest_strategy
    )
except ImportError:
    # 如果lib.evaluation导入失败，尝试直接导入
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
    from evaluation import (
        calculate_sharpe_ratio,
        calculate_volatility_adjusted_sharpe,
        evaluate_model,
        backtest_strategy
    )


def test_calculate_sharpe_ratio():
    """测试夏普比率计算"""
    
    # 测试正收益
    positive_returns = np.array([0.01, 0.02, 0.015, 0.008])
    sharpe = calculate_sharpe_ratio(positive_returns)
    assert sharpe > 0
    
    # 测试负收益
    negative_returns = np.array([-0.01, -0.02, -0.015, -0.008])
    sharpe = calculate_sharpe_ratio(negative_returns)
    assert sharpe < 0
    
    # 测试零收益
    zero_returns = np.array([0.0, 0.0, 0.0, 0.0])
    sharpe = calculate_sharpe_ratio(zero_returns)
    assert sharpe == 0.0
    
    # 测试无风险利率
    returns = np.array([0.01, 0.02, 0.015, 0.008])
    risk_free_rate = 0.005
    sharpe_with_rf = calculate_sharpe_ratio(returns, risk_free_rate)
    sharpe_without_rf = calculate_sharpe_ratio(returns)
    assert sharpe_with_rf != sharpe_without_rf


def test_calculate_volatility_adjusted_sharpe():
    """测试波动率调整的夏普比率"""
    
    returns = np.array([0.01, 0.02, -0.01, 0.015])
    market_volatility = 0.01  # 1% 市场波动率
    
    # 测试正常情况
    adjusted_sharpe = calculate_volatility_adjusted_sharpe(
        returns, 
        market_volatility,
        risk_free_rate=0.0
    )
    
    # 应该返回有效数值
    assert isinstance(adjusted_sharpe, float)
    
    # 测试高波动率惩罚
    high_vol_returns = np.array([0.05, -0.04, 0.06, -0.03])  # 高波动率
    adjusted_sharpe_high = calculate_volatility_adjusted_sharpe(
        high_vol_returns,
        market_volatility * 0.5,  # 市场波动率较低
        risk_free_rate=0.0,
        volatility_constraint=1.2
    )
    
    # 高波动率应该受到惩罚
    basic_sharpe = calculate_sharpe_ratio(high_vol_returns)
    assert adjusted_sharpe_high <= basic_sharpe


def test_evaluate_model():
    """测试模型评估"""
    
    # 创建测试数据
    y_true = np.array([0.01, 0.02, -0.01, 0.015])
    y_pred = np.array([0.012, 0.018, -0.008, 0.014])
    
    metrics = evaluate_model(y_true, y_pred)
    
    # 验证返回的指标
    expected_metrics = ['mse', 'rmse', 'mae', 'r2', 'sharpe_ratio', 'market_sharpe']
    for metric in expected_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], float)
    
    # 验证指标范围
    assert metrics['mse'] >= 0
    assert metrics['rmse'] >= 0
    assert metrics['mae'] >= 0
    assert metrics['r2'] <= 1  # R² 通常 <= 1


def test_backtest_strategy():
    """测试策略回测"""
    
    allocation = np.array([0.5, 1.2, 0.8, 1.0])  # 资产分配比例
    market_returns = np.array([0.01, -0.02, 0.03, -0.01])  # 市场收益
    
    metrics = backtest_strategy(allocation, market_returns)
    
    # 验证返回的指标
    expected_metrics = [
        'strategy_total_return', 'market_total_return', 'excess_return',
        'strategy_sharpe', 'market_sharpe', 'strategy_volatility', 
        'market_volatility', 'max_allocation', 'min_allocation', 'mean_allocation'
    ]
    
    for metric in expected_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], float)
    
    # 验证分配比例范围
    assert metrics['max_allocation'] <= 2.0
    assert metrics['min_allocation'] >= 0.0
    assert metrics['mean_allocation'] >= 0.0
    assert metrics['mean_allocation'] <= 2.0


def test_edge_cases():
    """测试边界情况"""
    
    # 测试空数组
    empty_returns = np.array([])
    sharpe = calculate_sharpe_ratio(empty_returns)
    assert sharpe == 0.0
    
    # 测试零标准差
    constant_returns = np.array([0.01, 0.01, 0.01, 0.01])
    sharpe = calculate_sharpe_ratio(constant_returns)
    assert sharpe == 0.0


if __name__ == "__main__":
    test_calculate_sharpe_ratio()
    test_calculate_volatility_adjusted_sharpe()
    test_evaluate_model()
    test_backtest_strategy()
    test_edge_cases()
    print("✅ 所有评估测试通过")