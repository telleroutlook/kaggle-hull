#!/usr/bin/env python3
"""
高级集成策略性能基准测试
比较不同集成方法的性能表现
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import warnings

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

# 抑制警告
warnings.filterwarnings('ignore')

# 导入必要的模块
try:
    from lib.models import (
        HullModel, 
        DynamicWeightedEnsemble,
        StackingEnsemble, 
        RiskAwareEnsemble,
        AveragingEnsemble,
        create_baseline_model,
        create_submission
    )
    from lib.features import FeaturePipeline
    from lib.evaluation import backtest_strategy
    from lib.utils import PerformanceTracker
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保在正确的目录中运行此脚本")
    sys.exit(1)


def create_synthetic_data(n_samples: int = 1000, n_features: int = 20) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """创建合成数据用于基准测试"""
    
    np.random.seed(42)
    
    # 创建特征
    features = {}
    for i in range(n_features):
        features[f'feature_{i}'] = np.random.randn(n_samples)
    
    # 创建目标变量（基于特征的线性组合 + 噪声）
    weights = np.random.randn(n_features)
    signal = sum(features[f'feature_{i}'] * weights[i] for i in range(n_features)) / n_features
    target = signal + 0.1 * np.random.randn(n_samples)
    
    # 创建市场收益率
    market_returns = 0.001 + 0.02 * np.random.randn(n_samples)  # 基础收益率 + 噪声
    
    return pd.DataFrame(features), pd.Series(target), pd.Series(market_returns)


def benchmark_ensemble_strategy(
    strategy_name: str,
    model_type: str,
    ensemble_config: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_runs: int = 3
) -> Dict[str, float]:
    """基准测试单个集成策略"""
    
    print(f"\n🔄 测试 {strategy_name}...")
    
    results = {
        'strategy_name': strategy_name,
        'model_type': model_type,
        'config': ensemble_config,
        'run_times': [],
        'prediction_stability': [],
        'sharpe_ratios': [],
        'volatility': [],
        'max_drawdown': []
    }
    
    for run in range(n_runs):
        try:
            start_time = time.time()
            
            # 创建模型
            if model_type == 'custom':
                # 使用自定义集成配置
                model = HullModel(
                    model_type=ensemble_config.get('base_type', 'ensemble'),
                    ensemble_config=ensemble_config
                )
            else:
                model = HullModel(model_type=model_type)
            
            # 训练
            model.fit(X_train, y_train)
            
            # 预测
            predictions = model.predict(X_test, clip=False)
            
            # 记录运行时间
            run_time = time.time() - start_time
            results['run_times'].append(run_time)
            
            # 评估预测稳定性
            pred_std = np.std(predictions)
            results['prediction_stability'].append(pred_std)
            
            # 模拟策略回测（简化版）
            if len(y_test) == len(predictions):
                # 将预测转换为分配
                allocations = np.clip(predictions, 0, 2)
                
                # 计算策略收益
                strategy_returns = allocations * y_test.values
                
                # 计算夏普比率
                mean_return = np.mean(strategy_returns)
                std_return = np.std(strategy_returns)
                sharpe = mean_return / std_return if std_return > 0 else 0
                results['sharpe_ratios'].append(sharpe)
                
                # 计算波动率
                results['volatility'].append(std_return)
                
                # 计算最大回撤
                cumulative = np.cumprod(1 + strategy_returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                results['max_drawdown'].append(np.min(drawdown))
            
            print(f"  ✅ 运行 {run+1}/{n_runs} 完成 - 时间: {run_time:.2f}s")
            
        except Exception as e:
            print(f"  ❌ 运行 {run+1} 失败: {e}")
            continue
    
    # 计算平均指标
    if results['run_times']:
        results['avg_runtime'] = np.mean(results['run_times'])
        results['std_runtime'] = np.std(results['run_times'])
        results['avg_prediction_stability'] = np.mean(results['prediction_stability'])
        results['avg_sharpe'] = np.mean(results['sharpe_ratios'])
        results['avg_volatility'] = np.mean(results['volatility'])
        results['avg_max_drawdown'] = np.mean(results['max_drawdown'])
    else:
        results['avg_runtime'] = float('inf')
        results['avg_prediction_stability'] = float('inf')
        results['avg_sharpe'] = -float('inf')
        results['avg_volatility'] = float('inf')
        results['avg_max_drawdown'] = -1.0
    
    return results


def run_ensemble_benchmark():
    """运行完整的集成策略基准测试"""
    
    print("🚀 开始高级集成策略性能基准测试")
    print("=" * 60)
    
    # 创建合成数据
    print("📊 生成测试数据...")
    X_train, y_train, market_returns_train = create_synthetic_data(800, 15)
    X_test, y_test, market_returns_test = create_synthetic_data(200, 15)
    
    print(f"  训练集大小: {len(X_train)}")
    print(f"  测试集大小: {len(X_test)}")
    print(f"  特征数量: {X_train.shape[1]}")
    
    # 定义要测试的策略
    strategies = [
        {
            'name': '基线集成 (均匀权重)',
            'type': 'ensemble',
            'config': {
                'base_type': 'ensemble',
                'weights': {'lightgbm': 1.0, 'xgboost': 1.0, 'catboost': 1.0}
            }
        },
        {
            'name': '动态权重集成',
            'type': 'custom',
            'config': {
                'base_type': 'dynamic_weighted_ensemble',
                'performance_window': 100,
                'conditional_weighting': True,
                'weight_smoothing': 0.1
            }
        },
        {
            'name': 'Stacking集成',
            'type': 'custom',
            'config': {
                'base_type': 'stacking_ensemble',
                'cv_folds': 3,
                'use_features_in_secondary': False
            }
        },
        {
            'name': '风险感知集成',
            'type': 'custom',
            'config': {
                'base_type': 'risk_aware_ensemble',
                'volatility_constraint': 1.2,
                'uncertainty_threshold': 0.1,
                'risk_parity': True
            }
        },
        {
            'name': 'LightGBM (基线)',
            'type': 'lightgbm',
            'config': {}
        },
        {
            'name': 'XGBoost (基线)',
            'type': 'xgboost',
            'config': {}
        }
    ]
    
    # 执行基准测试
    all_results = []
    
    for strategy in strategies:
        try:
            result = benchmark_ensemble_strategy(
                strategy_name=strategy['name'],
                model_type=strategy['type'],
                ensemble_config=strategy['config'],
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                n_runs=2  # 减少运行次数以节省时间
            )
            all_results.append(result)
        except Exception as e:
            print(f"❌ 策略 {strategy['name']} 测试失败: {e}")
            continue
    
    # 生成报告
    print("\n" + "=" * 60)
    print("📈 基准测试结果汇总")
    print("=" * 60)
    
    # 打印详细结果
    for result in all_results:
        if result['avg_runtime'] != float('inf'):
            print(f"\n🎯 {result['strategy_name']}")
            print(f"  平均运行时间: {result['avg_runtime']:.3f}±{result['std_runtime']:.3f}s")
            print(f"  预测稳定性: {result['avg_prediction_stability']:.4f}")
            print(f"  夏普比率: {result['avg_sharpe']:.4f}")
            print(f"  波动率: {result['avg_volatility']:.4f}")
            print(f"  最大回撤: {result['avg_max_drawdown']:.4f}")
        else:
            print(f"\n❌ {result['strategy_name']} - 测试失败")
    
    # 性能排名
    print(f"\n🏆 性能排名 (按夏普比率)")
    print("-" * 40)
    
    valid_results = [r for r in all_results if r['avg_sharpe'] != -float('inf')]
    if valid_results:
        valid_results.sort(key=lambda x: x['avg_sharpe'], reverse=True)
        
        for i, result in enumerate(valid_results, 1):
            print(f"{i}. {result['strategy_name']:<25} Sharpe: {result['avg_sharpe']:>7.4f}")
        
        # 最佳策略
        best_strategy = valid_results[0]
        baseline_results = [r for r in valid_results if 'LightGBM' in r['strategy_name'] or 'XGBoost' in r['strategy_name']]
        
        if baseline_results:
            baseline_sharpe = max(r['avg_sharpe'] for r in baseline_results)
            improvement = (best_strategy['avg_sharpe'] - baseline_sharpe) / abs(baseline_sharpe) * 100
            print(f"\n📊 最佳策略: {best_strategy['strategy_name']}")
            print(f"   相比基线提升: {improvement:+.1f}%")
    else:
        print("❌ 没有有效的测试结果")
    
    # 保存结果到文件
    results_file = Path("ensemble_benchmark_results.json")
    try:
        import json
        # 转换numpy类型为Python原生类型
        serializable_results = []
        for result in all_results:
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, (np.integer, np.floating)):
                    serializable_result[key] = float(value)
                elif isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                elif isinstance(value, dict):
                    serializable_result[key] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                              for k, v in value.items()}
                else:
                    serializable_result[key] = value
            serializable_results.append(serializable_result)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\n💾 结果已保存到: {results_file}")
        
    except Exception as e:
        print(f"⚠️ 保存结果失败: {e}")
    
    print("\n🎉 基准测试完成！")
    
    return all_results


if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    
    try:
        results = run_ensemble_benchmark()
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断测试")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
