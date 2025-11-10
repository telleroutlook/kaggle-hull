#!/usr/bin/env python3
"""
简化版集成策略性能基准测试
使用基础模型进行集成策略比较
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

def create_synthetic_data(n_samples: int = 500, n_features: int = 10) -> Tuple[pd.DataFrame, pd.Series]:
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
    
    return pd.DataFrame(features), pd.Series(target)


def test_simple_ensemble_strategies():
    """测试简单的集成策略（使用基础模型）"""
    
    print("🚀 简化版集成策略性能基准测试")
    print("=" * 50)
    
    # 创建数据
    print("📊 生成测试数据...")
    X_train, y_train = create_synthetic_data(400, 10)
    X_test, y_test = create_synthetic_data(100, 10)
    
    print(f"  训练集大小: {len(X_train)}")
    print(f"  测试集大小: {len(X_test)}")
    print(f"  特征数量: {X_train.shape[1]}")
    
    # 测试不同的简单集成策略
    results = []
    
    print("\n🔄 测试1: 单个基线模型")
    start_time = time.time()
    
    try:
        from lib.models import create_baseline_model
        
        model = create_baseline_model(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # 计算性能指标
        mse = np.mean((y_test.values - predictions) ** 2)
        mae = np.mean(np.abs(y_test.values - predictions))
        pred_std = np.std(predictions)
        
        single_model_time = time.time() - start_time
        
        results.append({
            'strategy': '单个基线模型',
            'mse': mse,
            'mae': mae,
            'prediction_std': pred_std,
            'runtime': single_model_time
        })
        
        print(f"  ✅ 完成 - MSE: {mse:.4f}, MAE: {mae:.4f}, 时间: {single_model_time:.2f}s")
        
    except Exception as e:
        print(f"  ❌ 失败: {e}")
    
    print("\n🔄 测试2: 简单平均集成 (3个基线模型)")
    start_time = time.time()
    
    try:
        models = []
        predictions_list = []
        
        for i in range(3):
            model = create_baseline_model(n_estimators=50, random_state=42+i)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            models.append(model)
            predictions_list.append(pred)
        
        # 简单平均
        ensemble_pred = np.mean(predictions_list, axis=0)
        
        # 计算性能指标
        mse = np.mean((y_test.values - ensemble_pred) ** 2)
        mae = np.mean(np.abs(y_test.values - ensemble_pred))
        pred_std = np.std(ensemble_pred)
        
        simple_ensemble_time = time.time() - start_time
        
        results.append({
            'strategy': '简单平均集成',
            'mse': mse,
            'mae': mae,
            'prediction_std': pred_std,
            'runtime': simple_ensemble_time
        })
        
        print(f"  ✅ 完成 - MSE: {mse:.4f}, MAE: {mae:.4f}, 时间: {simple_ensemble_time:.2f}s")
        
    except Exception as e:
        print(f"  ❌ 失败: {e}")
    
    print("\n🔄 测试3: 动态权重集成 (基于验证性能)")
    start_time = time.time()
    
    try:
        # 将训练数据分为训练和验证集
        val_size = int(0.2 * len(X_train))
        X_train_sub = X_train.iloc[:-val_size]
        y_train_sub = y_train.iloc[:-val_size]
        X_val = X_train.iloc[-val_size:]
        y_val = y_train.iloc[-val_size:]
        
        # 训练3个基线模型
        models = []
        val_predictions = []
        test_predictions = []
        
        for i in range(3):
            model = create_baseline_model(n_estimators=50, random_state=42+i)
            model.fit(X_train_sub, y_train_sub)
            
            # 验证集预测
            val_pred = model.predict(X_val)
            # 测试集预测
            test_pred = model.predict(X_test)
            
            models.append(model)
            val_predictions.append(val_pred)
            test_predictions.append(test_pred)
        
        # 基于验证性能计算权重
        val_mse = [np.mean((y_val.values - pred) ** 2) for pred in val_predictions]
        # 权重与性能成反比
        weights = [1.0 / (mse + 1e-8) for mse in val_mse]
        weights = np.array(weights)
        weights = weights / weights.sum()  # 归一化
        
        # 加权平均测试预测
        dynamic_pred = np.average(test_predictions, axis=0, weights=weights)
        
        # 计算性能指标
        mse = np.mean((y_test.values - dynamic_pred) ** 2)
        mae = np.mean(np.abs(y_test.values - dynamic_pred))
        pred_std = np.std(dynamic_pred)
        
        dynamic_ensemble_time = time.time() - start_time
        
        results.append({
            'strategy': '动态权重集成',
            'mse': mse,
            'mae': mae,
            'prediction_std': pred_std,
            'runtime': dynamic_ensemble_time,
            'weights': weights.tolist()
        })
        
        print(f"  ✅ 完成 - MSE: {mse:.4f}, MAE: {mae:.4f}, 时间: {dynamic_ensemble_time:.2f}s")
        print(f"     权重分配: {weights.round(3).tolist()}")
        
    except Exception as e:
        print(f"  ❌ 失败: {e}")
    
    print("\n🔄 测试4: 加权平均集成 (预设权重)")
    start_time = time.time()
    
    try:
        # 使用非均匀权重
        preset_weights = np.array([0.5, 0.3, 0.2])  # 第一个模型权重最高
        
        models = []
        predictions_list = []
        
        for i in range(3):
            model = create_baseline_model(n_estimators=50, random_state=42+i)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            models.append(model)
            predictions_list.append(pred)
        
        # 加权平均
        weighted_pred = np.average(predictions_list, axis=0, weights=preset_weights)
        
        # 计算性能指标
        mse = np.mean((y_test.values - weighted_pred) ** 2)
        mae = np.mean(np.abs(y_test.values - weighted_pred))
        pred_std = np.std(weighted_pred)
        
        weighted_ensemble_time = time.time() - start_time
        
        results.append({
            'strategy': '加权平均集成',
            'mse': mse,
            'mae': mae,
            'prediction_std': pred_std,
            'runtime': weighted_ensemble_time,
            'weights': preset_weights.tolist()
        })
        
        print(f"  ✅ 完成 - MSE: {mse:.4f}, MAE: {mae:.4f}, 时间: {weighted_ensemble_time:.2f}s")
        print(f"     权重分配: {preset_weights.tolist()}")
        
    except Exception as e:
        print(f"  ❌ 失败: {e}")
    
    # 结果汇总
    print("\n" + "=" * 50)
    print("📈 集成策略性能对比")
    print("=" * 50)
    
    if results:
        # 按MSE排序
        results.sort(key=lambda x: x['mse'])
        
        print(f"{'排名':<4} {'策略':<20} {'MSE':<10} {'MAE':<10} {'预测Std':<12} {'运行时间':<10}")
        print("-" * 70)
        
        for i, result in enumerate(results, 1):
            print(f"{i:<4} {result['strategy']:<20} {result['mse']:<10.4f} {result['mae']:<10.4f} "
                  f"{result['prediction_std']:<12.4f} {result['runtime']:<10.2f}s")
        
        # 最佳策略分析
        best = results[0]
        worst = results[-1]
        
        mse_improvement = (worst['mse'] - best['mse']) / worst['mse'] * 100
        mae_improvement = (worst['mae'] - best['mae']) / worst['mae'] * 100
        
        print(f"\n🏆 最佳策略: {best['strategy']}")
        print(f"   相比最差策略，MSE改善: {mse_improvement:.1f}%")
        print(f"   相比最差策略，MAE改善: {mae_improvement:.1f}%")
        
        # 简单统计
        single_model = next((r for r in results if r['strategy'] == '单个基线模型'), None)
        if single_model:
            best_mse = best['mse']
            single_mse = single_model['mse']
            if single_mse > 0:
                ensemble_improvement = (single_mse - best_mse) / single_mse * 100
                print(f"   相比单模型最佳集成改善: {ensemble_improvement:.1f}%")
        
        # 保存结果
        results_file = Path("simple_ensemble_benchmark.json")
        try:
            import json
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 结果已保存到: {results_file}")
        except Exception as e:
            print(f"\n⚠️ 保存结果失败: {e}")
        
    else:
        print("❌ 没有成功的测试结果")
    
    print("\n🎉 简化基准测试完成！")
    return results


if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    
    try:
        results = test_simple_ensemble_strategies()
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断测试")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)