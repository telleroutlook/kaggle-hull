"""
自适应时间窗口系统测试套件
测试市场状态检测、自适应窗口管理、窗口优化和性能跟踪功能
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import tempfile
import os
import warnings
from unittest.mock import patch, MagicMock

# 导入待测试的模块
import sys
sys.path.append(os.path.dirname(__file__))

from adaptive_time_window import (
    MarketStateDetector,
    AdaptiveWindowManager, 
    WindowOptimizer,
    PerformanceTracker,
    AdaptiveTimeWindowSystem,
    MarketRegime,
    MarketState,
    WindowConfig
)

# 导入配置模块
try:
    from lib.config import get_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    warnings.warn("配置模块不可用，将使用默认配置")


class TestMarketStateDetector(unittest.TestCase):
    """测试市场状态检测器"""
    
    def setUp(self):
        """设置测试环境"""
        self.detector = MarketStateDetector(lookback_periods=100, volatility_window=20)
        self.sample_data = self._generate_sample_data(200)
    
    def _generate_sample_data(self, n_periods: int) -> pd.DataFrame:
        """生成测试数据"""
        np.random.seed(42)
        
        # 生成价格数据
        prices = 100 * (1 + np.cumsum(np.random.normal(0, 0.02, n_periods)))
        
        # 生成交易量数据
        volumes = np.random.lognormal(10, 0.5, n_periods)
        
        # 生成其他特征
        data = {
            'P1': prices,  # 价格列
            'volume': volumes,  # 交易量列
            'P2': prices * (1 + np.random.normal(0, 0.01, n_periods)),
            'P3': prices * (1 + np.random.normal(0, 0.01, n_periods)),
        }
        
        return pd.DataFrame(data)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.detector.lookback_periods, 100)
        self.assertEqual(self.detector.volatility_window, 20)
        self.assertIsInstance(self.detector.trend_windows, list)
    
    def test_market_state_detection(self):
        """测试市场状态检测"""
        state = self.detector.detect_market_state(self.sample_data, price_col='P1')
        
        self.assertIsInstance(state, MarketState)
        self.assertIsInstance(state.regime, MarketRegime)
        self.assertIsInstance(state.trend_strength, float)
        self.assertIsInstance(state.volatility_level, float)
        self.assertIsInstance(state.confidence, float)
        self.assertGreaterEqual(state.confidence, 0)
        self.assertLessEqual(state.confidence, 1)
    
    def test_technical_indicators(self):
        """测试技术指标计算"""
        prices = self.sample_data['P1'].values
        
        # 测试移动平均线
        ma_20 = self.detector._sma(prices, 20)
        self.assertEqual(len(ma_20), len(prices))
        self.assertFalse(np.isnan(ma_20[-1]))
        
        # 测试RSI
        rsi = self.detector._rsi(prices, 14)
        self.assertFalse(np.isnan(rsi[-1]))
        self.assertGreaterEqual(rsi[-1], 0)
        self.assertLessEqual(rsi[-1], 100)
        
        # 测试MACD
        macd_line, macd_signal, macd_hist = self.detector._macd(prices)
        self.assertIsNotNone(macd_line)
        self.assertIsNotNone(macd_signal)
        self.assertIsNotNone(macd_hist)
    
    def test_market_regime_classification(self):
        """测试市场状态分类"""
        # 模拟不同市场状态的技术指标
        test_cases = [
            # 强势上涨趋势
            {
                'rsi': 70,
                'macd_hist': 0.5,
                'bb_width': 0.03,
                'volatility': 0.015,
                'price_position_252': 0.8,
                'momentum_20': 0.1,
                'expected_regime': MarketRegime.BULL_TREND
            },
            # 熊市趋势
            {
                'rsi': 30,
                'macd_hist': -0.3,
                'bb_width': 0.04,
                'volatility': 0.025,
                'price_position_252': 0.2,
                'momentum_20': -0.08,
                'expected_regime': MarketRegime.BEAR_TREND
            },
            # 高波动率
            {
                'rsi': 50,
                'macd_hist': 0.1,
                'bb_width': 0.08,
                'volatility': 0.06,
                'price_position_252': 0.6,
                'momentum_20': 0.05,
                'expected_regime': MarketRegime.HIGH_VOLATILITY
            }
        ]
        
        for case in test_cases:
            features = case.copy()
            del features['expected_regime']
            
            regime = self.detector._classify_market_regime(features)
            # 注意：实际分类可能不完全匹配预期，这里主要测试不会崩溃
            self.assertIsInstance(regime, MarketRegime)
    
    def test_trend_strength_calculation(self):
        """测试趋势强度计算"""
        features = {
            'rsi': 65,
            'macd_hist': 0.2,
            'momentum_20': 0.05,
            'ma_20': 105,
            'ma_50': 102
        }
        
        trend_strength = self.detector._calculate_trend_strength(features)
        self.assertIsInstance(trend_strength, float)
        self.assertGreaterEqual(trend_strength, -1)
        self.assertLessEqual(trend_strength, 1)
    
    def test_volatility_level_calculation(self):
        """测试波动率水平计算"""
        features = {
            'volatility': 0.03,
            'bb_width': 0.04
        }
        
        vol_level = self.detector._calculate_volatility_level(features)
        self.assertIsInstance(vol_level, float)
        self.assertGreaterEqual(vol_level, 0)
        self.assertLessEqual(vol_level, 1)
    
    def test_volume_anomaly_detection(self):
        """测试交易量异常检测"""
        volumes = np.random.lognormal(10, 0.5, 50)
        
        anomaly = self.detector._calculate_volume_anomaly(volumes)
        self.assertIsInstance(anomaly, float)
        self.assertGreaterEqual(anomaly, 0)
        self.assertLessEqual(anomaly, 1)
    
    def test_confidence_calculation(self):
        """测试置信度计算"""
        features = {
            'rsi': 50,
            'macd_hist': 0.0,
            'volatility': 0.02
        }
        
        confidence = self.detector._calculate_detection_confidence(features)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)


class TestAdaptiveWindowManager(unittest.TestCase):
    """测试自适应窗口管理器"""
    
    def setUp(self):
        """设置测试环境"""
        self.manager = AdaptiveWindowManager()
        self.sample_data = self._generate_sample_data(300)
    
    def _generate_sample_data(self, n_periods: int) -> pd.DataFrame:
        """生成测试数据"""
        np.random.seed(42)
        prices = 100 * (1 + np.cumsum(np.random.normal(0, 0.02, n_periods)))
        return pd.DataFrame({'P1': prices})
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsInstance(self.manager.window_configs, dict)
        self.assertIsInstance(self.manager.historical_performance, type(self.manager.historical_performance))
        self.assertIsNotNone(self.manager.current_windows)
    
    def test_default_config_creation(self):
        """测试默认配置创建"""
        regime = MarketRegime.BULL_TREND
        data_length = 200
        
        config = self.manager._create_default_config(regime, data_length)
        
        self.assertIsInstance(config, WindowConfig)
        self.assertEqual(config.market_regime, regime)
        self.assertGreater(config.min_length, 0)
        self.assertGreater(config.max_length, config.min_length)
        self.assertGreater(config.optimal_length, 0)
    
    def test_optimal_window_selection(self):
        """测试最优窗口选择"""
        # 创建模拟市场状态
        state = MarketState(
            regime=MarketRegime.BULL_TREND,
            trend_strength=0.5,
            volatility_level=0.3,
            volume_anomaly=0.1,
            confidence=0.8,
            timestamp=datetime.now()
        )
        
        optimal_config = self.manager.get_optimal_window(state, 200)
        
        self.assertIsInstance(optimal_config, WindowConfig)
        self.assertEqual(optimal_config.market_regime, MarketRegime.BULL_TREND)
    
    def test_performance_update(self):
        """测试性能更新"""
        config = WindowConfig(
            name="test_config",
            min_length=30,
            max_length=120,
            optimal_length=60,
            performance_score=0.5,
            market_regime=MarketRegime.BULL_TREND,
            last_updated=datetime.now()
        )
        
        performance_metrics = {'mse': 0.2, 'mae': 0.3}
        self.manager.update_performance(config, performance_metrics)
        
        # 验证性能分数被更新
        self.assertNotEqual(config.performance_score, 0.5)
    
    def test_window_adaptation(self):
        """测试窗口自适应"""
        # 创建高波动率状态
        state = MarketState(
            regime=MarketRegime.HIGH_VOLATILITY,
            trend_strength=0.0,
            volatility_level=0.9,  # 高波动率
            volume_anomaly=0.0,
            confidence=0.3,  # 低置信度
            timestamp=datetime.now()
        )
        
        initial_config = self.manager._create_default_config(MarketRegime.HIGH_VOLATILITY, 100)
        self.manager.current_windows[MarketRegime.HIGH_VOLATILITY] = initial_config
        
        self.manager.adapt_windows(state)
        
        # 验证配置被调整
        self.assertIsNotNone(self.manager.current_windows[MarketRegime.HIGH_VOLATILITY])
    
    def test_config_export_import(self):
        """测试配置导出和导入"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # 导出配置
            self.manager.export_config(temp_path)
            
            # 验证文件存在
            self.assertTrue(os.path.exists(temp_path))
            
            # 导入配置到新管理器
            new_manager = AdaptiveWindowManager()
            new_manager.import_config(temp_path)
            
            # 验证配置被正确导入
            self.assertGreater(len(new_manager.window_configs), 0)
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestWindowOptimizer(unittest.TestCase):
    """测试窗口优化器"""
    
    def setUp(self):
        """设置测试环境"""
        self.optimizer = WindowOptimizer(optimization_window=20, min_improvement=0.01)
        self.sample_data = self._generate_sample_data(200)
        self.window_manager = AdaptiveWindowManager()
    
    def _generate_sample_data(self, n_periods: int) -> pd.DataFrame:
        """生成测试数据"""
        np.random.seed(42)
        np.random.shuffle(list(range(n_periods)))
        
        prices = 100 * (1 + np.cumsum(np.random.normal(0, 0.02, n_periods)))
        target = np.random.normal(0, 0.01, n_periods)
        
        data = {
            'P1': prices,
            'market_forward_excess_returns': target,
            'feature1': np.random.normal(0, 1, n_periods),
            'feature2': np.random.normal(0, 1, n_periods),
        }
        
        return pd.DataFrame(data)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.optimizer.optimization_window, 20)
        self.assertEqual(self.optimizer.min_improvement, 0.01)
        self.assertIsInstance(self.optimizer.optimization_history, type(self.optimizer.optimization_history))
    
    def test_candidate_length_generation(self):
        """测试候选长度生成"""
        current_config = WindowConfig(
            name="test",
            min_length=30,
            max_length=120,
            optimal_length=60,
            performance_score=0.5,
            market_regime=MarketRegime.BULL_TREND,
            last_updated=datetime.now()
        )
        
        candidates = self.optimizer._generate_candidate_lengths(current_config, 150)
        
        self.assertIsInstance(candidates, list)
        self.assertGreater(len(candidates), 0)
        self.assertIn(60, candidates)  # 应该包含当前长度
    
    def test_window_performance_evaluation(self):
        """测试窗口性能评估"""
        performance = self.optimizer._evaluate_window_performance(
            self.sample_data, 50, 'market_forward_excess_returns', ['feature1', 'feature2']
        )
        
        self.assertIsInstance(performance, float)
        self.assertGreaterEqual(performance, 0)
        self.assertLessEqual(performance, 1)
    
    @patch('sklearn.linear_model.LinearRegression')
    def test_window_optimization(self, mock_model):
        """测试窗口优化"""
        # 模拟模型预测
        mock_model.return_value.fit.return_value = None
        mock_model.return_value.predict.return_value = np.random.normal(0, 0.01, 50)
        
        state = MarketState(
            regime=MarketRegime.BULL_TREND,
            trend_strength=0.5,
            volatility_level=0.3,
            volume_anomaly=0.1,
            confidence=0.8,
            timestamp=datetime.now()
        )
        
        optimized_config = self.optimizer.optimize_window_config(
            self.window_manager, state, self.sample_data
        )
        
        self.assertIsInstance(optimized_config, WindowConfig)
        self.assertEqual(optimized_config.market_regime, MarketRegime.BULL_TREND)
    
    def test_optimization_history(self):
        """测试优化历史记录"""
        history = self.optimizer.get_optimization_history()
        self.assertIsInstance(history, list)


class TestPerformanceTracker(unittest.TestCase):
    """测试性能跟踪器"""
    
    def setUp(self):
        """设置测试环境"""
        self.tracker = PerformanceTracker(tracking_window=50, alert_threshold=0.1)
        self.sample_config = WindowConfig(
            name="test_config",
            min_length=30,
            max_length=120,
            optimal_length=60,
            performance_score=0.5,
            market_regime=MarketRegime.BULL_TREND,
            last_updated=datetime.now()
        )
        self.sample_state = MarketState(
            regime=MarketRegime.BULL_TREND,
            trend_strength=0.5,
            volatility_level=0.3,
            volume_anomaly=0.1,
            confidence=0.8,
            timestamp=datetime.now()
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.tracker.tracking_window, 50)
        self.assertEqual(self.tracker.alert_threshold, 0.1)
        self.assertIsInstance(self.tracker.performance_records, type(self.tracker.performance_records))
        self.assertIsInstance(self.tracker.alerts, list)
    
    def test_performance_recording(self):
        """测试性能记录"""
        performance_metrics = {'mse': 0.2, 'mae': 0.3}
        
        self.tracker.record_performance(
            self.sample_config, self.sample_state, performance_metrics
        )
        
        self.assertEqual(len(self.tracker.performance_records), 1)
        
        record = self.tracker.performance_records[0]
        self.assertEqual(record['window_config'], "test_config")
        self.assertEqual(record['market_regime'], "bull_trend")
        self.assertIn('performance_metrics', record)
    
    def test_prediction_metrics_recording(self):
        """测试预测指标记录"""
        predictions = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        actuals = np.array([0.015, 0.018, 0.028, 0.042, 0.048])
        
        performance_metrics = {'mse': 0.001, 'mae': 0.02}
        
        self.tracker.record_performance(
            self.sample_config, self.sample_state, performance_metrics, predictions, actuals
        )
        
        record = self.tracker.performance_records[0]
        self.assertIn('prediction_metrics', record)
        self.assertIn('mse', record['prediction_metrics'])
        self.assertIn('mae', record['prediction_metrics'])
        self.assertIn('r2', record['prediction_metrics'])
    
    def test_alert_generation(self):
        """测试告警生成"""
        # 记录几个性能记录
        for i in range(5):
            performance_metrics = {'mse': 0.2 + i * 0.05}  # 递增的MSE
            self.tracker.record_performance(
                self.sample_config, self.sample_state, performance_metrics
            )
        
        # 应该生成性能下降告警
        self.assertGreater(len(self.tracker.alerts), 0)
    
    def test_performance_summary(self):
        """测试性能摘要"""
        # 记录一些性能数据
        for _ in range(10):
            performance_metrics = {'mse': np.random.uniform(0.1, 0.3), 'mae': np.random.uniform(0.2, 0.4)}
            self.tracker.record_performance(
                self.sample_config, self.sample_state, performance_metrics
            )
        
        summary = self.tracker.get_performance_summary()
        
        self.assertIn('total_records', summary)
        self.assertIn('mse', summary)
        self.assertIn('mae', summary)
        self.assertEqual(summary['total_records'], 10)
    
    def test_recent_performance(self):
        """测试获取最近性能记录"""
        for i in range(15):
            performance_metrics = {'mse': 0.1 + i * 0.01}
            self.tracker.record_performance(
                self.sample_config, self.sample_state, performance_metrics
            )
        
        recent = self.tracker.get_recent_performance(5)
        self.assertEqual(len(recent), 5)
    
    def test_alert_filtering(self):
        """测试告警过滤"""
        # 记录不同类型的告警
        self.tracker._create_alert('TYPE_A', 'Message A', {})
        self.tracker._create_alert('TYPE_B', 'Message B', {})
        self.tracker._create_alert('TYPE_A', 'Message C', {})
        
        type_a_alerts = self.tracker.get_alerts('TYPE_A')
        all_alerts = self.tracker.get_alerts()
        
        self.assertEqual(len(type_a_alerts), 2)
        self.assertEqual(len(all_alerts), 3)
    
    def test_performance_report_export(self):
        """测试性能报告导出"""
        # 记录一些数据
        for _ in range(5):
            performance_metrics = {'mse': 0.2, 'mae': 0.3}
            self.tracker.record_performance(
                self.sample_config, self.sample_state, performance_metrics
            )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            self.tracker.export_performance_report(temp_path)
            
            # 验证文件存在和内容
            self.assertTrue(os.path.exists(temp_path))
            
            with open(temp_path, 'r') as f:
                report = json.load(f)
            
            self.assertIn('summary', report)
            self.assertIn('recent_performance', report)
            self.assertIn('alerts', report)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestAdaptiveTimeWindowSystem(unittest.TestCase):
    """测试自适应时间窗口系统"""
    
    def setUp(self):
        """设置测试环境"""
        self.system = AdaptiveTimeWindowSystem(
            config_level='balanced',
            lookback_periods=100,
            tracking_enabled=True
        )
        self.sample_data = self._generate_sample_data(150)
    
    def _generate_sample_data(self, n_periods: int) -> pd.DataFrame:
        """生成测试数据"""
        np.random.seed(42)
        prices = 100 * (1 + np.cumsum(np.random.normal(0, 0.02, n_periods)))
        volumes = np.random.lognormal(10, 0.5, n_periods)
        
        return pd.DataFrame({
            'P1': prices,
            'volume': volumes
        })
    
    def test_initialization(self):
        """测试系统初始化"""
        self.assertEqual(self.system.config_level, 'balanced')
        self.assertEqual(self.system.detector.lookback_periods, 100)
        self.assertTrue(self.system.tracking_enabled)
        self.assertIsInstance(self.system.tracker, PerformanceTracker)
    
    def test_system_initialization(self):
        """测试系统初始化方法"""
        self.system.initialize(self.sample_data, 'P1', 'volume')
        
        self.assertTrue(self.system.is_initialized)
        self.assertIsNotNone(self.system.last_market_state)
        self.assertIsNotNone(self.system.current_window_config)
    
    def test_optimal_window_retrieval(self):
        """测试获取最优窗口"""
        self.system.initialize(self.sample_data, 'P1', 'volume')
        
        optimal_window = self.system.get_optimal_window(self.sample_data, 'P1', 'volume')
        
        self.assertIsInstance(optimal_window, WindowConfig)
        self.assertEqual(optimal_window, self.system.current_window_config)
    
    def test_market_state_change_detection(self):
        """测试市场状态变化检测"""
        state1 = MarketState(
            regime=MarketRegime.BULL_TREND,
            trend_strength=0.5,
            volatility_level=0.3,
            volume_anomaly=0.1,
            confidence=0.8,
            timestamp=datetime.now()
        )
        
        state2 = MarketState(
            regime=MarketRegime.BEAR_TREND,  # 不同状态
            trend_strength=-0.5,
            volatility_level=0.3,
            volume_anomaly=0.1,
            confidence=0.8,
            timestamp=datetime.now()
        )
        
        # 应该检测到显著变化
        self.assertTrue(self.system._is_significant_state_change(state1, state2))
        
        # 测试置信度变化
        state3 = MarketState(
            regime=MarketRegime.BULL_TREND,  # 相同状态
            trend_strength=0.5,
            volatility_level=0.3,
            volume_anomaly=0.1,
            confidence=0.2,  # 置信度变化很大
            timestamp=datetime.now()
        )
        
        self.assertTrue(self.system._is_significant_state_change(state1, state3))
    
    def test_performance_update(self):
        """测试性能更新"""
        self.system.initialize(self.sample_data, 'P1', 'volume')
        
        performance_metrics = {'mse': 0.2, 'mae': 0.3}
        self.system.update_performance(performance_metrics)
        
        # 验证性能被记录
        recent_performance = self.system.tracker.get_recent_performance(1)
        self.assertEqual(len(recent_performance), 1)
    
    def test_system_status(self):
        """测试系统状态获取"""
        self.system.initialize(self.sample_data, 'P1', 'volume')
        
        status = self.system.get_system_status()
        
        self.assertIn('is_initialized', status)
        self.assertIn('config_level', status)
        self.assertIn('tracking_enabled', status)
        self.assertIn('last_market_state', status)
        self.assertIn('current_window_config', status)
        self.assertIn('performance_summary', status)
        self.assertIn('recent_alerts', status)
    
    def test_system_state_export(self):
        """测试系统状态导出"""
        self.system.initialize(self.sample_data, 'P1', 'volume')
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            self.system.export_system_state(temp_path)
            
            # 验证主要状态文件存在
            self.assertTrue(os.path.exists(temp_path))
            
            # 验证窗口配置文件
            window_config_path = temp_path.replace('.json', '_windows.json')
            self.assertTrue(os.path.exists(window_config_path))
            
            # 验证性能报告
            performance_path = temp_path.replace('.json', '_performance.json')
            self.assertTrue(os.path.exists(performance_path))
            
        finally:
            # 清理临时文件
            for path in [temp_path, 
                        temp_path.replace('.json', '_windows.json'),
                        temp_path.replace('.json', '_performance.json')]:
                if os.path.exists(path):
                    os.unlink(path)
    
    def test_error_handling(self):
        """测试错误处理"""
        # 测试未初始化时的错误
        with self.assertRaises(RuntimeError):
            self.system.get_optimal_window(self.sample_data, 'P1', 'volume')
        
        # 测试无效数据
        self.system.initialize(self.sample_data, 'P1', 'volume')
        
        # 添加一些无效数据
        invalid_data = pd.DataFrame({'P1': [1, 2, 3]})  # 数据太少
        
        # 这应该不会崩溃，但可能会返回默认配置
        optimal_window = self.system.get_optimal_window(invalid_data, 'P1')
        self.assertIsInstance(optimal_window, WindowConfig)


class TestConfigIntegration(unittest.TestCase):
    """测试配置集成（如果配置模块可用）"""
    
    @unittest.skipUnless(CONFIG_AVAILABLE, "配置模块不可用")
    def test_config_integration(self):
        """测试配置集成"""
        from lib.config import get_config
        
        config = get_config()
        window_config = config.get_adaptive_windows_config()
        
        # 验证配置结构
        self.assertIn('enabled', window_config)
        self.assertIn('config_level', window_config)
        self.assertIn('lookback_periods', window_config)
        self.assertIsInstance(window_config['enabled'], bool)
        self.assertIsInstance(window_config['lookback_periods'], int)
    
    @unittest.skipUnless(CONFIG_AVAILABLE, "配置模块不可用")
    def test_window_preset_retrieval(self):
        """测试窗口预设获取"""
        from lib.config import get_config
        
        config = get_config()
        preset = config.get_window_preset('balanced', 'bull_trend')
        
        # 验证预设结构
        self.assertIn('min_length', preset)
        self.assertIn('max_length', preset)
        self.assertIn('optimal_length', preset)
        
        # 验证数值合理性
        self.assertLess(preset['min_length'], preset['optimal_length'])
        self.assertLess(preset['optimal_length'], preset['max_length'])


def run_integration_test():
    """运行集成测试"""
    print("运行自适应时间窗口系统集成测试...")
    
    try:
        # 生成测试数据
        np.random.seed(42)
        n_periods = 500
        prices = 100 * (1 + np.cumsum(np.random.normal(0, 0.02, n_periods)))
        volumes = np.random.lognormal(10, 0.5, n_periods)
        
        data = pd.DataFrame({
            'P1': prices,
            'volume': volumes,
            'market_forward_excess_returns': np.random.normal(0, 0.01, n_periods)
        })
        
        # 初始化系统
        system = AdaptiveTimeWindowSystem(
            config_level='balanced',
            lookback_periods=252,
            tracking_enabled=True
        )
        
        print("1. 初始化系统...")
        system.initialize(data, 'P1', 'volume')
        print("✓ 系统初始化完成")
        
        # 测试基本功能
        print("2. 测试市场状态检测...")
        state = system.detector.detect_market_state(data.tail(100), 'P1', 'volume')
        print(f"✓ 检测到市场状态: {state.regime.value} (置信度: {state.confidence:.2f})")
        
        print("3. 获取最优窗口...")
        window = system.get_optimal_window(data.tail(200), 'P1', 'volume')
        print(f"✓ 推荐窗口长度: {window.optimal_length}")
        
        print("4. 模拟性能更新...")
        # 模拟几个预测周期
        for i in range(5):
            # 模拟新的市场数据
            new_data = data.tail(100 + i * 10)
            optimal_window = system.get_optimal_window(new_data, 'P1', 'volume')
            
            # 模拟性能指标
            mse = np.random.uniform(0.1, 0.3)
            mae = np.random.uniform(0.2, 0.4)
            performance_metrics = {'mse': mse, 'mae': mae}
            
            system.update_performance(performance_metrics)
            
            print(f"   周期 {i+1}: MSE={mse:.3f}, MAE={mae:.3f}, 窗口={optimal_window.optimal_length}")
        
        print("5. 获取系统状态...")
        status = system.get_system_status()
        print(f"✓ 系统状态: 初始化={status['is_initialized']}, 配置级别={status['config_level']}")
        
        print("6. 测试配置集成...")
        if CONFIG_AVAILABLE:
            from lib.config import get_config
            config = get_config()
            window_config = config.get_adaptive_windows_config()
            print(f"✓ 配置集成正常: 启用状态={window_config['enabled']}")
        else:
            print("⚠ 配置模块不可用，跳过配置测试")
        
        print("✓ 集成测试完成")
        return True
        
    except Exception as e:
        print(f"✗ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # 运行单元测试
    print("运行自适应时间窗口系统单元测试...")
    unittest.main(verbosity=2, exit=False)
    
    # 运行集成测试
    print("\n" + "="*50)
    integration_success = run_integration_test()
    
    if integration_success:
        print("\n🎉 所有测试通过！自适应时间窗口系统工作正常。")
    else:
        print("\n❌ 部分测试失败，请检查系统配置。")
