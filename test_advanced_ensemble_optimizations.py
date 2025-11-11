#!/usr/bin/env python3
"""
高级集成策略优化测试脚本
测试所有新增的集成策略和功能
"""

import sys
import os
sys.path.append('/home/dev/github/kaggle-hull')

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 导入优化后的模型模块
try:
    from working.lib.models import (
        AdvancedModelPerformanceMonitor,
        AdvancedConditionalWeightEngine,
        DynamicWeightedEnsemble,
        RiskAwareEnsemble,
        AdversarialEnsemble,
        MultiLevelEnsemble,
        RealTimePerformanceMonitor,
        create_lightgbm_model,
        create_xgboost_model,
        create_catboost_model,
    )
    print("✅ 成功导入所有优化后的模型类")
    
    # 导入基线模型
    try:
        from working.lib.models import create_baseline_model
    except ImportError as e:
        print(f"⚠️ 基线模型导入失败: {e}")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)


def create_test_data(n_samples=1000, n_features=20, random_state=42):
    """创建测试数据"""
    X, y = make_regression(
        n_samples=n_samples, 
        n_features=n_features, 
        noise=0.1, 
        random_state=random_state
    )
    
    # 添加一些非线性特征
    X[:, 0] = X[:, 0] ** 2
    X[:, 1] = np.sin(X[:, 1])
    
    # 创建DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    return X_df, y_series


def test_advanced_performance_monitor():
    """测试高级性能监控器"""
    print("\n🔍 测试 AdvancedModelPerformanceMonitor")
    
    monitor = AdvancedModelPerformanceMonitor(
        window_size=50, 
        update_frequency=5,
        market_regime_detection=True
    )
    
    # 模拟性能数据
    true_values = np.random.normal(0, 1, 100)
    predictions = np.random.normal(0, 1, 100)
    
    # 更新性能
    for i in range(0, 100, 5):
        monitor.update_performance(
            f"model_1", 
            true_values[i:i+5], 
            predictions[i:i+5], 
            i
        )
    
    # 获取权重
    weights = monitor.get_model_weights(["model_1"], market_regime_aware=True)
    print(f"   性能权重: {weights}")
    
    # 检查市场状态检测
    regime = monitor._detect_market_regime(true_values[-20:])
    print(f"   检测到的市场状态: {regime}")
    
    print("✅ 高级性能监控器测试通过")


def test_advanced_conditional_weight_engine():
    """测试高级条件化权重引擎"""
    print("\n🔍 测试 AdvancedConditionalWeightEngine")
    
    engine = AdvancedConditionalWeightEngine(market_lookback=30)
    
    # 模拟市场数据
    market_returns = np.random.normal(0, 0.02, 50)
    model_performances = {"lightgbm": 0.1, "xgboost": 0.12, "catboost": 0.15}
    
    # 更新市场状态
    state = engine.update_market_state(market_returns, model_performances)
    print(f"   市场状态: {state}")
    
    # 获取条件化权重
    weights = engine.get_conditional_weights(state)
    print(f"   条件化权重: {weights}")
    
    # 检查状态转移概率
    trans_probs = engine.get_state_transition_probability(state)
    print(f"   状态转移概率: {trans_probs}")
    
    print("✅ 高级条件化权重引擎测试通过")


def test_dynamic_weighted_ensemble():
    """测试动态权重集成器"""
    print("\n🔍 测试 DynamicWeightedEnsemble")
    
    # 创建基础模型 - 适配可用依赖
    models = []
    try:
        models.append(create_lightgbm_model(random_state=42, n_estimators=100))
    except ImportError:
        print("   ⚠️ LightGBM 不可用，跳过")
    
    try:
        models.append(create_xgboost_model(random_state=42, n_estimators=100))
    except ImportError:
        print("   ⚠️ XGBoost 不可用，跳过")
    
    try:
        models.append(create_catboost_model(random_state=42, iterations=100))
    except ImportError:
        print("   ⚠️ CatBoost 不可用，跳过")
    
    if not models:
        # 使用基线模型作为备选
        from working.lib.models import create_baseline_model
        models = [create_baseline_model(random_state=42, n_estimators=100) for _ in range(3)]
        print("   使用基线模型作为备选")
    
    # 创建集成器
    ensemble = DynamicWeightedEnsemble(
        models,
        performance_window=30,
        conditional_weighting=True,
        weight_smoothing=0.1
    )
    
    # 准备数据
    X, y = create_test_data(n_samples=500)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练
    ensemble.fit(X_train, y_train)
    print("   ✅ 训练完成")
    
    # 预测
    predictions = ensemble.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"   测试 MSE: {mse:.6f}")
    print(f"   测试 MAE: {mae:.6f}")
    print(f"   当前权重: {ensemble.current_weights}")
    
    print("✅ 动态权重集成器测试通过")


def test_risk_aware_ensemble():
    """测试风险感知集成器"""
    print("\n🔍 测试 RiskAwareEnsemble")
    
    models = []
    try:
        models.append(create_lightgbm_model(random_state=42, n_estimators=50))
    except ImportError:
        pass
    
    try:
        models.append(create_xgboost_model(random_state=42, n_estimators=50))
    except ImportError:
        pass
    
    if not models:
        from working.lib.models import create_baseline_model
        models = [create_baseline_model(random_state=42, n_estimators=50) for _ in range(2)]
    
    ensemble = RiskAwareEnsemble(
        models,
        dynamic_risk_adjustment=True
    )
    
    X, y = create_test_data(n_samples=300)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练
    ensemble.fit(X_train, y_train)
    print("   ✅ 训练完成")
    
    # 预测
    predictions, uncertainties = ensemble.predict(X_test, return_uncertainty=True)
    
    print(f"   平均不确定性: {np.mean(uncertainties):.6f}")
    print(f"   不确定性范围: [{np.min(uncertainties):.6f}, {np.max(uncertainties):.6f}]")
    
    print("✅ 风险感知集成器测试通过")


def test_adversarial_ensemble():
    """测试对抗性集成器"""
    print("\n🔍 测试 AdversarialEnsemble")
    
    models = []
    try:
        models.append(create_lightgbm_model(random_state=42, n_estimators=50))
    except ImportError:
        pass
    
    try:
        models.append(create_xgboost_model(random_state=42, n_estimators=50))
    except ImportError:
        pass
    
    if not models:
        from working.lib.models import create_baseline_model
        models = [create_baseline_model(random_state=42, n_estimators=50) for _ in range(2)]
    
    ensemble = AdversarialEnsemble(
        models,
        adversarial_ratio=0.1,
        noise_std=0.01
    )
    
    X, y = create_test_data(n_samples=200)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练
    ensemble.fit(X_train, y_train)
    print("   ✅ 训练完成")
    
    # 预测
    predictions = ensemble.predict(X_test)
    
    print(f"   鲁棒性分数: {dict(ensemble.model_robustness_scores)}")
    
    print("✅ 对抗性集成器测试通过")


def test_multi_level_ensemble():
    """测试多层次集成器"""
    print("\n🔍 测试 MultiLevelEnsemble")
    
    # 创建不同层次的模型
    level1_models = []
    level2_models = []
    
    try:
        level1_models.append(create_lightgbm_model(random_state=42, n_estimators=50))
    except ImportError:
        pass
    
    try:
        level2_models.append(create_xgboost_model(random_state=42, n_estimators=50))
    except ImportError:
        pass
    
    # 如果没有足够模型，使用基线模型
    if not level1_models:
        level1_models = [create_baseline_model(random_state=42, n_estimators=50)]
    
    if not level2_models:
        level2_models = [create_baseline_model(random_state=43, n_estimators=50)]
    
    # 创建简单元学习器
    from sklearn.linear_model import Ridge
    meta_ensemble = Ridge(alpha=1.0)
    
    ensemble = MultiLevelEnsemble(
        level1_models,
        level2_models,
        meta_ensemble=meta_ensemble
    )
    
    X, y = create_test_data(n_samples=300)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练
    ensemble.fit(X_train, y_train)
    print("   ✅ 训练完成")
    
    # 预测
    predictions = ensemble.predict(X_test)
    
    print(f"   Level1权重: {ensemble.level1_weights}")
    print(f"   Level2权重: {ensemble.level2_weights}")
    print(f"   混合权重: {ensemble.meta_weights}")
    
    print("✅ 多层次集成器测试通过")


def test_realtime_performance_monitor():
    """测试实时性能监控器"""
    print("\n🔍 测试 RealTimePerformanceMonitor")
    
    monitor = RealTimePerformanceMonitor(
        health_check_interval=5,
        failure_threshold=0.2
    )
    
    # 模拟模型性能评估
    baseline_perf = 0.1
    
    for i in range(10):
        # 模拟性能变化
        recent_perf = baseline_perf + np.random.normal(0, 0.1)
        
        health_status = monitor.assess_model_health(
            f"model_{i % 3}", 
            recent_perf, 
            baseline_perf
        )
        
        print(f"   模型 {i % 3} 状态: {health_status}, 健康分数: {monitor.model_health_scores[f'model_{i % 3}']:.3f}")
    
    # 获取健康状态总结
    summary = monitor.get_health_summary()
    print(f"   健康状态总结: {summary}")
    
    print("✅ 实时性能监控器测试通过")


def benchmark_ensemble_performance():
    """集成策略性能基准测试"""
    print("\n🏆 集成策略性能基准测试")
    
    # 创建数据集
    X, y = create_test_data(n_samples=800)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 基础模型 - 适配可用依赖
    models = []
    
    try:
        models.append(create_lightgbm_model(random_state=42, n_estimators=100))
    except ImportError:
        print("   ⚠️ LightGBM 不可用")
    
    try:
        models.append(create_xgboost_model(random_state=42, n_estimators=100))
    except ImportError:
        print("   ⚠️ XGBoost 不可用")
    
    try:
        models.append(create_catboost_model(random_state=42, iterations=100, verbose=False))
    except ImportError:
        print("   ⚠️ CatBoost 不可用")
    
    if not models:
        # 使用基线模型
        from working.lib.models import create_baseline_model
        models = [create_baseline_model(random_state=42, n_estimators=100) for _ in range(3)]
        print("   使用基线模型进行测试")
    
    results = {}
    
    # 1. 动态权重集成
    print("   测试动态权重集成...")
    dynamic_ensemble = DynamicWeightedEnsemble(models, conditional_weighting=True)
    dynamic_ensemble.fit(X_train, y_train)
    dynamic_pred = dynamic_ensemble.predict(X_test)
    results['DynamicWeighted'] = {
        'mse': mean_squared_error(y_test, dynamic_pred),
        'mae': mean_absolute_error(y_test, dynamic_pred)
    }
    
    # 2. 风险感知集成
    print("   测试风险感知集成...")
    risk_ensemble = RiskAwareEnsemble(models, dynamic_risk_adjustment=True)
    risk_ensemble.fit(X_train, y_train)
    risk_pred, _ = risk_ensemble.predict(X_test, return_uncertainty=True)
    results['RiskAware'] = {
        'mse': mean_squared_error(y_test, risk_pred),
        'mae': mean_absolute_error(y_test, risk_pred)
    }
    
    # 3. 对抗性集成
    print("   测试对抗性集成...")
    adv_ensemble = AdversarialEnsemble(models, adversarial_ratio=0.1)
    adv_ensemble.fit(X_train, y_train)
    adv_pred = adv_ensemble.predict(X_test)
    results['Adversarial'] = {
        'mse': mean_squared_error(y_test, adv_pred),
        'mae': mean_absolute_error(y_test, adv_pred)
    }
    
    # 4. 简单平均
    print("   测试简单平均集成...")
    from working.lib.models import AveragingEnsemble
    simple_ensemble = AveragingEnsemble(models)
    simple_ensemble.fit(X_train, y_train)
    simple_pred = simple_ensemble.predict(X_test)
    results['SimpleAverage'] = {
        'mse': mean_squared_error(y_test, simple_pred),
        'mae': mean_absolute_error(y_test, simple_pred)
    }
    
    # 打印结果
    print("\n📊 性能基准测试结果:")
    print("策略               MSE          MAE")
    print("-" * 40)
    for strategy, metrics in results.items():
        print(f"{strategy:<18} {metrics['mse']:.6f}    {metrics['mae']:.6f}")
    
    # 计算改进
    baseline_mse = results['SimpleAverage']['mse']
    for strategy, metrics in results.items():
        if strategy != 'SimpleAverage':
            improvement = (baseline_mse - metrics['mse']) / baseline_mse * 100
            print(f"   {strategy} 相对改进: {improvement:+.2f}%")
    
    return results


def main():
    """主测试函数"""
    print("🚀 开始测试高级集成策略优化")
    print("=" * 50)
    
    try:
        # 1. 测试各个组件
        test_advanced_performance_monitor()
        test_advanced_conditional_weight_engine()
        test_dynamic_weighted_ensemble()
        test_risk_aware_ensemble()
        test_adversarial_ensemble()
        test_multi_level_ensemble()
        test_realtime_performance_monitor()
        
        # 2. 性能基准测试
        benchmark_results = benchmark_ensemble_performance()
        
        print("\n" + "=" * 50)
        print("✅ 所有测试完成！")
        print("\n🔧 优化总结:")
        print("1. ✅ 动态权重算法改进 - 添加时间窗口自适应和市场状态感知")
        print("2. ✅ 风险感知集成机制 - 增强不确定性和波动率感知")
        print("3. ✅ 高级集成策略 - 实现对抗性、多层次集成")
        print("4. ✅ 实时性能监控 - 添加健康状态检测和自动故障处理")
        
        return benchmark_results
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
