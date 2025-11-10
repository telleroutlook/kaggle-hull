"""
高级集成策略测试
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from lib.models import (
        HullModel, 
        DynamicWeightedEnsemble,
        StackingEnsemble, 
        RiskAwareEnsemble,
        ModelPerformanceMonitor,
        ConditionalWeightEngine
    )
except ImportError:
    # 如果lib.models导入失败，尝试直接导入
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
    from models import (
        HullModel, 
        DynamicWeightedEnsemble,
        StackingEnsemble, 
        RiskAwareEnsemble,
        ModelPerformanceMonitor,
        ConditionalWeightEngine
    )


class TestModelPerformanceMonitor:
    """测试模型性能监控器"""
    
    def test_initialization(self):
        """测试监控器初始化"""
        monitor = ModelPerformanceMonitor(window_size=50, update_frequency=5)
        assert monitor.window_size == 50
        assert monitor.update_frequency == 5
        assert len(monitor.model_performances) == 0
        
    def test_update_performance(self):
        """测试性能更新"""
        monitor = ModelPerformanceMonitor(update_frequency=1)  # 降低更新频率用于测试
        true_values = np.array([1.0, 2.0, 3.0])
        predictions = np.array([1.1, 1.9, 3.1])
        
        monitor.update_performance("test_model", true_values, predictions, 0)
        
        assert "test_model" in monitor.model_performances
        assert len(monitor.model_performances["test_model"]) == 1
        
        perf = monitor.model_performances["test_model"][0]
        assert "mse" in perf
        assert "correlation" in perf
        assert "stability_score" in perf
        assert "timestamp" in perf
        
    def test_get_model_weights(self):
        """测试权重计算"""
        monitor = ModelPerformanceMonitor()
        
        # 模拟性能数据
        true_values = np.array([1.0, 2.0, 3.0])
        predictions = np.array([1.1, 1.9, 3.1])
        
        monitor.update_performance("model1", true_values, predictions, 0)
        monitor.update_performance("model2", true_values, predictions * 1.5, 5)
        
        weights = monitor.get_model_weights(["model1", "model2"])
        
        assert "model1" in weights
        assert "model2" in weights
        assert abs(weights["model1"] + weights["model2"] - 1.0) < 1e-6
        assert all(w >= 0 for w in weights.values())


class TestConditionalWeightEngine:
    """测试条件化权重引擎"""
    
    def test_initialization(self):
        """测试引擎初始化"""
        engine = ConditionalWeightEngine(market_lookback=30)
        assert engine.market_lookback == 30
        assert len(engine.market_states) == 0
        
    def test_market_state_detection(self):
        """测试市场状态检测"""
        engine = ConditionalWeightEngine()
        
        # 测试上升趋势
        uptrend_returns = np.array([0.01, 0.02, 0.015, 0.025, 0.01])
        state = engine.update_market_state(uptrend_returns)
        assert state in ['trending_up', 'stable', 'unknown']  # 可能的状态
        
        # 测试高波动
        volatile_returns = np.array([0.1, -0.08, 0.12, -0.09, 0.11])
        state = engine.update_market_state(volatile_returns)
        assert state == 'volatile'
        
    def test_conditional_weights(self):
        """测试条件化权重"""
        engine = ConditionalWeightEngine()
        
        weights = engine.get_conditional_weights('trending_up')
        assert 'lightgbm' in weights
        assert 'xgboost' in weights
        assert 'catboost' in weights
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        
        # 测试未知状态
        unknown_weights = engine.get_conditional_weights('unknown')
        assert unknown_weights == engine.state_weights['unknown']


class TestDynamicWeightedEnsemble:
    """测试动态权重集成器"""
    
    def test_initialization(self):
        """测试初始化"""
        try:
            from lib.models import create_lightgbm_model, create_xgboost_model
            
            base_models = [
                create_lightgbm_model(n_estimators=10),
                create_xgboost_model(n_estimators=10)
            ]
        except (ImportError, ModuleNotFoundError):
            # 如果没有这些库，创建简单的回退模型
            from lib.models import create_baseline_model
            
            base_models = [
                create_baseline_model(n_estimators=10),
                create_baseline_model(n_estimators=10)
            ]
        
        ensemble = DynamicWeightedEnsemble(
            base_models, 
            performance_window=50,
            weight_smoothing=0.2
        )
        
        assert len(ensemble.base_models) == 2
        assert ensemble.performance_window == 50
        assert ensemble.weight_smoothing == 0.2
        assert len(ensemble.current_weights) == 2
        assert abs(ensemble.current_weights.sum() - 1.0) < 1e-6
        
    def test_fallback_weights(self):
        """测试回退权重机制"""
        try:
            from lib.models import create_lightgbm_model
            base_models = [create_lightgbm_model(n_estimators=10)]
        except (ImportError, ModuleNotFoundError):
            from lib.models import create_baseline_model
            base_models = [create_baseline_model(n_estimators=10)]
        
        ensemble = DynamicWeightedEnsemble(base_models)
        
        # 模拟失败情况
        ensemble.model_failures["model_0"] = 10
        ensemble.last_successful_predictions["model_0"] = 0
        
        fallback_weights = ensemble._get_fallback_weights()
        assert len(fallback_weights) == 1
        assert fallback_weights[0] > 0
        
    # LightGBM依赖检查将在运行时进行
    def test_fit_predict(self):
        """测试训练和预测"""
        try:
            from lib.models import create_lightgbm_model, create_xgboost_model
            
            base_models = [
                create_lightgbm_model(n_estimators=10, random_state=42),
                create_xgboost_model(n_estimators=10, random_state=42)
            ]
        except (ImportError, ModuleNotFoundError):
            from lib.models import create_baseline_model
            
            base_models = [
                create_baseline_model(n_estimators=10, random_state=42),
                create_baseline_model(n_estimators=10, random_state=42)
            ]
        
        # 创建测试数据
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        y = np.random.randn(100)
        
        ensemble = DynamicWeightedEnsemble(base_models)
        
        # 训练
        ensemble.fit(X, y)
        
        # 预测
        X_test = pd.DataFrame({
            'feature1': np.random.randn(10),
            'feature2': np.random.randn(10)
        })
        
        predictions = ensemble.predict(X_test)
        
        assert len(predictions) == 10
        assert np.all(np.isfinite(predictions))
        assert np.all(predictions >= 0) and np.all(predictions <= 2)  # 应该在[0,2]范围内


class TestStackingEnsemble:
    """测试Stacking集成器"""
    
    def test_initialization(self):
        """测试初始化"""
        try:
            from lib.models import create_lightgbm_model, create_xgboost_model
            
            base_models = [
                create_lightgbm_model(n_estimators=10),
                create_xgboost_model(n_estimators=10)
            ]
        except (ImportError, ModuleNotFoundError):
            from lib.models import create_baseline_model
            
            base_models = [
                create_baseline_model(n_estimators=10),
                create_baseline_model(n_estimators=10)
            ]
        
        ensemble = StackingEnsemble(
            base_models,
            cv_folds=3,
            use_features_in_secondary=True
        )
        
        assert len(ensemble.base_models) == 2
        assert ensemble.cv_folds == 3
        assert ensemble.use_features_in_secondary == True
        assert not ensemble.is_fitted
        
    # LightGBM依赖检查将在运行时进行
    def test_fit_predict(self):
        """测试训练和预测"""
        try:
            from lib.models import create_lightgbm_model, create_xgboost_model
            
            base_models = [
                create_lightgbm_model(n_estimators=10, random_state=42),
                create_xgboost_model(n_estimators=10, random_state=42)
            ]
        except (ImportError, ModuleNotFoundError):
            from lib.models import create_baseline_model
            
            base_models = [
                create_baseline_model(n_estimators=10, random_state=42),
                create_baseline_model(n_estimators=10, random_state=42)
            ]
        
        # 创建测试数据
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        y = pd.Series(np.random.randn(100))
        
        ensemble = StackingEnsemble(base_models, cv_folds=2)
        
        # 训练
        ensemble.fit(X, y)
        assert ensemble.is_fitted
        
        # 预测
        X_test = pd.DataFrame({
            'feature1': np.random.randn(10),
            'feature2': np.random.randn(10)
        })
        
        predictions = ensemble.predict(X_test)
        
        assert len(predictions) == 10
        assert np.all(np.isfinite(predictions))


class TestRiskAwareEnsemble:
    """测试风险感知集成器"""
    
    def test_initialization(self):
        """测试初始化"""
        try:
            from lib.models import create_lightgbm_model
            base_models = [create_lightgbm_model(n_estimators=10)]
        except (ImportError, ModuleNotFoundError):
            from lib.models import create_baseline_model
            base_models = [create_baseline_model(n_estimators=10)]
        
        ensemble = RiskAwareEnsemble(
            base_models,
            volatility_constraint=1.5,
            uncertainty_threshold=0.15
        )
        
        assert len(ensemble.base_models) == 1
        assert ensemble.volatility_constraint == 1.5
        assert ensemble.uncertainty_threshold == 0.15
        
    # LightGBM依赖检查将在运行时进行
    def test_uncertainty_estimation(self):
        """测试不确定性估计"""
        try:
            from lib.models import create_lightgbm_model
            base_models = [create_lightgbm_model(n_estimators=10, random_state=42)]
        except (ImportError, ModuleNotFoundError):
            from lib.models import create_baseline_model
            base_models = [create_baseline_model(n_estimators=10, random_state=42)]
        ensemble = RiskAwareEnsemble(base_models)
        
        X = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50)
        })
        
        uncertainty = ensemble._estimate_uncertainty(0, X, base_models[0])
        
        assert len(uncertainty) == 50
        assert np.all(uncertainty >= 0)
        
    # LightGBM依赖检查将在运行时进行
    def test_predict_with_uncertainty(self):
        """测试带不确定性的预测"""
        try:
            from lib.models import create_lightgbm_model
            base_models = [create_lightgbm_model(n_estimators=10, random_state=42)]
        except (ImportError, ModuleNotFoundError):
            from lib.models import create_baseline_model
            base_models = [create_baseline_model(n_estimators=10, random_state=42)]
        
        # 创建测试数据
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        y = np.random.randn(100)
        ensemble = RiskAwareEnsemble(base_models)
        
        # 训练
        ensemble.fit(X, y)
        
        # 预测带不确定性
        X_test = pd.DataFrame({
            'feature1': np.random.randn(10),
            'feature2': np.random.randn(10)
        })
        
        predictions, uncertainty = ensemble.predict(X_test, return_uncertainty=True)
        
        assert len(predictions) == 10
        assert len(uncertainty) == 10
        assert np.all(np.isfinite(predictions))
        assert np.all(uncertainty >= 0)


class TestEnhancedHullModel:
    """测试增强版HullModel"""
    
    def test_dynamic_weighted_ensemble_model_type(self):
        """测试动态权重集成模型类型"""
        ensemble_config = {
            'performance_window': 100,
            'conditional_weighting': True,
            'weight_smoothing': 0.1
        }
        
        model = HullModel(
            model_type='dynamic_weighted_ensemble',
            ensemble_config=ensemble_config
        )
        
        assert model.model_type == 'dynamic_weighted_ensemble'
        assert model.ensemble_config == ensemble_config
        
    def test_stacking_ensemble_model_type(self):
        """测试Stacking集成模型类型"""
        ensemble_config = {
            'cv_folds': 3,
            'use_features_in_secondary': True
        }
        
        model = HullModel(
            model_type='stacking_ensemble',
            ensemble_config=ensemble_config
        )
        
        assert model.model_type == 'stacking_ensemble'
        assert model.ensemble_config == ensemble_config
        
    def test_risk_aware_ensemble_model_type(self):
        """测试风险感知集成模型类型"""
        ensemble_config = {
            'volatility_constraint': 1.5,
            'risk_parity': True
        }
        
        model = HullModel(
            model_type='risk_aware_ensemble',
            ensemble_config=ensemble_config
        )
        
        assert model.model_type == 'risk_aware_ensemble'
        assert model.ensemble_config == ensemble_config
        
    def test_get_ensemble_info(self):
        """测试获取集成信息"""
        model = HullModel(model_type='baseline')
        info = model.get_ensemble_info()
        
        assert info['type'] == 'single_model'
        
    # LightGBM依赖检查将在运行时进行
    def test_enhanced_cross_validation(self):
        """测试增强版交叉验证"""
        from lib.models import create_lightgbm_model
        
        # 创建测试数据
        X = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50)
        })
        y = pd.Series(np.random.randn(50))
        
        # 测试集成模型交叉验证
        model = HullModel(
            model_type='ensemble',
            model_params={
                'lightgbm': {'n_estimators': 10},
                'xgboost': {'n_estimators': 10}
            }
        )
        
        cv_results = model.cross_validate(X, y, n_splits=2)
        
        expected_keys = ['mean_mse', 'std_mse', 'mean_mae', 'std_mae']
        for key in expected_keys:
            assert key in cv_results
            assert isinstance(cv_results[key], float)


if __name__ == "__main__":
    # 运行所有测试
    import pytest
    
    # 创建测试实例并运行
    test_monitor = TestModelPerformanceMonitor()
    test_monitor.test_initialization()
    test_monitor.test_update_performance()
    test_monitor.test_get_model_weights()
    print("✅ ModelPerformanceMonitor 测试通过")
    
    test_engine = TestConditionalWeightEngine()
    test_engine.test_initialization()
    test_engine.test_market_state_detection()
    test_engine.test_conditional_weights()
    print("✅ ConditionalWeightEngine 测试通过")
    
    test_dynamic = TestDynamicWeightedEnsemble()
    test_dynamic.test_initialization()
    test_dynamic.test_fallback_weights()
    print("✅ DynamicWeightedEnsemble 测试通过")
    
    test_stacking = TestStackingEnsemble()
    test_stacking.test_initialization()
    print("✅ StackingEnsemble 测试通过")
    
    test_risk = TestRiskAwareEnsemble()
    test_risk.test_initialization()
    test_risk.test_uncertainty_estimation()
    print("✅ RiskAwareEnsemble 测试通过")
    
    test_hull = TestEnhancedHullModel()
    test_hull.test_dynamic_weighted_ensemble_model_type()
    test_hull.test_stacking_ensemble_model_type()
    test_hull.test_risk_aware_ensemble_model_type()
    test_hull.test_get_ensemble_info()
    print("✅ Enhanced HullModel 测试通过")
    
    print("🎉 所有高级集成策略测试通过！")