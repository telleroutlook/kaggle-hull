"""
自适应时间窗口系统
根据市场条件动态调整预测窗口大小，提升预测准确性
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
import warnings

# 技术指标计算
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    warnings.warn("TA-Lib not available, using fallback implementations")

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats


class MarketRegime(Enum):
    """市场状态枚举"""
    BULL_TREND = "bull_trend"        # 牛市趋势
    BEAR_TREND = "bear_trend"        # 熊市趋势
    SIDEWAYS = "sideways"            # 横盘震荡
    HIGH_VOLATILITY = "high_vol"     # 高波动
    LOW_VOLATILITY = "low_vol"       # 低波动
    BREAKOUT = "breakout"            # 突破
    CRISIS = "crisis"                # 危机/极端情况
    NORMAL = "normal"                # 正常市场


@dataclass
class MarketState:
    """市场状态数据类"""
    regime: MarketRegime
    trend_strength: float            # 趋势强度 [-1, 1]
    volatility_level: float          # 波动率水平 [0, 1]
    volume_anomaly: float            # 交易量异常度 [0, 1]
    confidence: float                # 状态检测置信度 [0, 1]
    timestamp: datetime
    features: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'regime': self.regime.value,
            'trend_strength': self.trend_strength,
            'volatility_level': self.volatility_level,
            'volume_anomaly': self.volume_anomaly,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'features': self.features
        }


@dataclass
class WindowConfig:
    """窗口配置数据类"""
    name: str
    min_length: int
    max_length: int
    optimal_length: int
    performance_score: float
    market_regime: MarketRegime
    last_updated: datetime
    usage_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'min_length': self.min_length,
            'max_length': self.max_length,
            'optimal_length': self.optimal_length,
            'performance_score': self.performance_score,
            'market_regime': self.market_regime.value,
            'last_updated': self.last_updated.isoformat(),
            'usage_count': self.usage_count
        }


class MarketStateDetector:
    """市场状态检测器"""
    
    def __init__(self, 
                 lookback_periods: int = 252,
                 volatility_window: int = 20,
                 trend_windows: List[int] = None,
                 volume_window: int = 20):
        """
        初始化市场状态检测器
        
        Args:
            lookback_periods: 回看期间长度
            volatility_window: 波动率计算窗口
            trend_windows: 趋势检测窗口列表
            volume_window: 交易量分析窗口
        """
        self.lookback_periods = lookback_periods
        self.volatility_window = volatility_window
        self.trend_windows = trend_windows or [5, 10, 20, 50]
        self.volume_window = volume_window
        self.logger = logging.getLogger(__name__)
        
        # 缓存技术指标计算结果
        self._indicator_cache = {}
        
    def detect_market_state(self, 
                           data: pd.DataFrame, 
                           price_col: str = 'P1',
                           volume_col: str = None) -> MarketState:
        """
        检测当前市场状态
        
        Args:
            data: 包含价格和交易量数据的数据框
            price_col: 价格列名
            volume_col: 交易量列名（可选）
            
        Returns:
            MarketState: 市场状态对象
        """
        try:
            # 获取必要的数据
            prices = data[price_col].values
            if volume_col and volume_col in data.columns:
                volumes = data[volume_col].values
            else:
                volumes = np.ones_like(prices)  # 如果没有交易量数据，使用全1数组
            
            # 计算各种技术指标
            features = self._calculate_technical_indicators(prices, volumes)
            
            # 分析市场状态
            regime = self._classify_market_regime(features)
            trend_strength = self._calculate_trend_strength(features)
            volatility_level = self._calculate_volatility_level(features)
            volume_anomaly = self._calculate_volume_anomaly(volumes)
            confidence = self._calculate_detection_confidence(features)
            
            return MarketState(
                regime=regime,
                trend_strength=trend_strength,
                volatility_level=volatility_level,
                volume_anomaly=volume_anomaly,
                confidence=confidence,
                timestamp=datetime.now(),
                features=features
            )
            
        except Exception as e:
            self.logger.error(f"市场状态检测失败: {e}")
            return MarketState(
                regime=MarketRegime.NORMAL,
                trend_strength=0.0,
                volatility_level=0.5,
                volume_anomaly=0.0,
                confidence=0.0,
                timestamp=datetime.now(),
                features={}
            )
    
    def _calculate_technical_indicators(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, float]:
        """计算技术指标"""
        features = {}
        
        try:
            # 移动平均线
            for window in self.trend_windows:
                if len(prices) >= window:
                    ma = self._sma(prices, window)
                    if not np.isnan(ma[-1]):
                        features[f'ma_{window}'] = ma[-1]
            
            # RSI
            if len(prices) >= 14:
                rsi = self._rsi(prices, 14)
                if not np.isnan(rsi[-1]):
                    features['rsi'] = rsi[-1]
            
            # MACD
            if len(prices) >= 26:
                macd_line, macd_signal, macd_hist = self._macd(prices)
                if not np.isnan(macd_line[-1]):
                    features['macd'] = macd_line[-1]
                if not np.isnan(macd_hist[-1]):
                    features['macd_hist'] = macd_hist[-1]
            
            # 布林带
            if len(prices) >= 20:
                bb_upper, bb_middle, bb_lower = self._bollinger_bands(prices, 20, 2)
                if not np.isnan(bb_upper[-1]):
                    features['bb_width'] = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1]
                    features['bb_position'] = (prices[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
            
            # 波动率
            if len(prices) >= self.volatility_window:
                volatility = self._calculate_volatility(prices, self.volatility_window)
                features['volatility'] = volatility[-1] if not np.isnan(volatility[-1]) else 0.0
            
            # 交易量指标
            if len(volumes) >= self.volume_window:
                volume_sma = self._sma(volumes, self.volume_window)
                if not np.isnan(volume_sma[-1]) and volume_sma[-1] > 0:
                    features['volume_ratio'] = volumes[-1] / volume_sma[-1]
            
            # 价格动量
            for period in [1, 5, 10, 20]:
                if len(prices) >= period:
                    momentum = (prices[-1] - prices[-period]) / prices[-period]
                    features[f'momentum_{period}'] = momentum
            
            # 价格位置
            if len(prices) >= 252:
                price_position = (prices[-1] - np.min(prices[-252:])) / (np.max(prices[-252:]) - np.min(prices[-252:]))
                features['price_position_252'] = price_position
            
        except Exception as e:
            self.logger.warning(f"技术指标计算部分失败: {e}")
        
        return features
    
    def _classify_market_regime(self, features: Dict[str, float]) -> MarketRegime:
        """分类市场状态"""
        try:
            # 获取关键指标
            rsi = features.get('rsi', 50)
            macd_hist = features.get('macd_hist', 0)
            bb_width = features.get('bb_width', 0.02)
            volatility = features.get('volatility', 0.02)
            price_position = features.get('price_position_252', 0.5)
            momentum_20 = features.get('momentum_20', 0)
            
            # 趋势判断
            is_uptrend = (
                rsi > 50 and 
                macd_hist > 0 and 
                momentum_20 > 0 and 
                price_position > 0.6
            )
            
            is_downtrend = (
                rsi < 50 and 
                macd_hist < 0 and 
                momentum_20 < 0 and 
                price_position < 0.4
            )
            
            # 波动率判断
            high_vol = volatility > 0.03  # 3%日波动率
            low_vol = volatility < 0.01   # 1%日波动率
            
            # 异常情况判断
            if volatility > 0.05 or bb_width > 0.1:  # 极高波动率或布林带极宽
                return MarketRegime.CRISIS
            
            if high_vol and abs(momentum_20) > 0.1:  # 高波动且大动量
                return MarketRegime.BREAKOUT
            
            if is_uptrend:
                return MarketRegime.BULL_TREND if not high_vol else MarketRegime.HIGH_VOLATILITY
            
            if is_downtrend:
                return MarketRegime.BEAR_TREND if not high_vol else MarketRegime.HIGH_VOLATILITY
            
            if low_vol:
                return MarketRegime.LOW_VOLATILITY
            
            return MarketRegime.SIDEWAYS
            
        except Exception as e:
            self.logger.warning(f"市场状态分类失败: {e}")
            return MarketRegime.NORMAL
    
    def _calculate_trend_strength(self, features: Dict[str, float]) -> float:
        """计算趋势强度"""
        try:
            # 基于RSI、MACD和动量计算趋势强度
            rsi = features.get('rsi', 50) / 100.0  # 标准化到[0,1]
            macd_hist = features.get('macd_hist', 0)
            momentum_20 = features.get('momentum_20', 0)
            
            # 移动平均线斜率
            ma_slope = 0
            for window in [10, 20, 50]:
                ma_key = f'ma_{window}'
                if ma_key in features:
                    # 这里简化处理，实际应该计算斜率
                    ma_slope += features[ma_key]
            
            # 综合趋势强度 [-1, 1]
            trend_score = (
                (rsi - 0.5) * 2 +  # RSI偏离50的程度
                np.tanh(macd_hist * 100) +  # MACD柱状图
                np.tanh(momentum_20 * 10) +  # 动量
                np.tanh(ma_slope / window) * 0.1  # 移动平均线影响
            ) / 4
            
            return np.clip(trend_score, -1, 1)
            
        except Exception as e:
            self.logger.warning(f"趋势强度计算失败: {e}")
            return 0.0
    
    def _calculate_volatility_level(self, features: Dict[str, float]) -> float:
        """计算波动率水平"""
        try:
            volatility = features.get('volatility', 0.02)
            bb_width = features.get('bb_width', 0.02)
            
            # 标准化波动率 [0, 1]
            # 假设正常波动率在1-3%之间
            vol_level = np.clip((volatility - 0.005) / 0.025, 0, 1)
            
            # 结合布林带宽度
            bb_level = np.clip((bb_width - 0.01) / 0.04, 0, 1)
            
            return (vol_level + bb_level) / 2
            
        except Exception as e:
            self.logger.warning(f"波动率水平计算失败: {e}")
            return 0.5
    
    def _calculate_volume_anomaly(self, volumes: np.ndarray) -> float:
        """计算交易量异常度"""
        try:
            if len(volumes) < self.volume_window * 2:
                return 0.0
            
            # 计算交易量分位数
            recent_vol = volumes[-self.volume_window:]
            historical_vol = volumes[-self.volume_window*2:-self.volume_window]
            
            recent_mean = np.mean(recent_vol)
            historical_mean = np.mean(historical_vol)
            
            if historical_mean > 0:
                volume_ratio = recent_vol[-1] / historical_mean
                # 转换为[0,1]的异常度
                anomaly = np.clip((volume_ratio - 1) / 2, 0, 1)
                return anomaly
            
            return 0.0
            
        except Exception as e:
            self.logger.warning(f"交易量异常度计算失败: {e}")
            return 0.0
    
    def _calculate_detection_confidence(self, features: Dict[str, float]) -> float:
        """计算检测置信度"""
        try:
            # 基于特征完整性和一致性计算置信度
            required_features = ['rsi', 'macd', 'volatility']
            available_count = sum(1 for f in required_features if f in features)
            feature_completeness = available_count / len(required_features)
            
            # 指标一致性
            consistency_score = 1.0
            if 'rsi' in features and 'macd_hist' in features:
                rsi = features['rsi']
                macd = features['macd_hist']
                # 如果RSI和MACD方向一致，增加置信度
                if (rsi > 50 and macd > 0) or (rsi < 50 and macd < 0):
                    consistency_score = 0.9
                else:
                    consistency_score = 0.6
            
            return feature_completeness * consistency_score
            
        except Exception as e:
            self.logger.warning(f"置信度计算失败: {e}")
            return 0.5
    
    # 技术指标计算方法（使用TA-Lib或fallback实现）
    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """简单移动平均"""
        if TALIB_AVAILABLE:
            return talib.SMA(data, timeperiod=period)
        else:
            return pd.Series(data).rolling(window=period).mean().values
    
    def _rsi(self, data: np.ndarray, period: int) -> np.ndarray:
        """RSI指标"""
        if TALIB_AVAILABLE:
            return talib.RSI(data, timeperiod=period)
        else:
            delta = pd.Series(data).diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs)).values
    
    def _macd(self, data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD指标"""
        if TALIB_AVAILABLE:
            macd_line = talib.MACD(data, fastperiod=fast, slowperiod=slow, signalperiod=signal)[0]
            macd_signal = talib.MACD(data, fastperiod=fast, slowperiod=slow, signalperiod=signal)[1]
            macd_hist = talib.MACD(data, fastperiod=fast, slowperiod=slow, signalperiod=signal)[2]
        else:
            ema_fast = pd.Series(data).ewm(span=fast).mean()
            ema_slow = pd.Series(data).ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            macd_signal = macd_line.ewm(span=signal).mean()
            macd_hist = macd_line - macd_signal
            return macd_line.values, macd_signal.values, macd_hist.values
        
        return macd_line, macd_signal, macd_hist
    
    def _bollinger_bands(self, data: np.ndarray, period: int = 20, std_dev: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """布林带"""
        if TALIB_AVAILABLE:
            upper, middle, lower = talib.BBANDS(data, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
        else:
            rolling_mean = pd.Series(data).rolling(window=period).mean()
            rolling_std = pd.Series(data).rolling(window=period).std()
            upper = rolling_mean + (rolling_std * std_dev)
            middle = rolling_mean
            lower = rolling_mean - (rolling_std * std_dev)
            return upper.values, middle.values, lower.values
        
        return upper, middle, lower
    
    def _calculate_volatility(self, data: np.ndarray, period: int) -> np.ndarray:
        """计算波动率"""
        returns = pd.Series(data).pct_change()
        rolling_vol = returns.rolling(window=period).std() * np.sqrt(252)  # 年化
        return rolling_vol.values


class AdaptiveWindowManager:
    """自适应窗口管理器"""
    
    def __init__(self, 
                 base_config: Optional[Dict[str, Any]] = None,
                 memory_size: int = 1000,
                 adaptation_threshold: float = 0.1):
        """
        初始化自适应窗口管理器
        
        Args:
            base_config: 基础窗口配置
            memory_size: 历史配置记忆大小
            adaptation_threshold: 自适应阈值
        """
        self.base_config = base_config or self._get_default_config()
        self.memory_size = memory_size
        self.adaptation_threshold = adaptation_threshold
        
        # 窗口配置存储
        self.window_configs: Dict[MarketRegime, List[WindowConfig]] = defaultdict(list)
        self.historical_performance: deque = deque(maxlen=memory_size)
        self.current_windows: Dict[MarketRegime, WindowConfig] = {}
        
        # 初始化默认配置
        self._initialize_default_configs()
        
        self.logger = logging.getLogger(__name__)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'conservative': {
                MarketRegime.BULL_TREND: {'min': 120, 'max': 252, 'optimal': 180},
                MarketRegime.BEAR_TREND: {'min': 90, 'max': 180, 'optimal': 120},
                MarketRegime.SIDEWAYS: {'min': 30, 'max': 90, 'optimal': 60},
                MarketRegime.HIGH_VOLATILITY: {'min': 45, 'max': 120, 'optimal': 75},
                MarketRegime.LOW_VOLATILITY: {'min': 120, 'max': 252, 'optimal': 180},
                MarketRegime.BREAKOUT: {'min': 20, 'max': 60, 'optimal': 40},
                MarketRegime.CRISIS: {'min': 10, 'max': 30, 'optimal': 20},
                MarketRegime.NORMAL: {'min': 60, 'max': 180, 'optimal': 120}
            },
            'balanced': {
                MarketRegime.BULL_TREND: {'min': 90, 'max': 180, 'optimal': 120},
                MarketRegime.BEAR_TREND: {'min': 60, 'max': 120, 'optimal': 90},
                MarketRegime.SIDEWAYS: {'min': 20, 'max': 60, 'optimal': 40},
                MarketRegime.HIGH_VOLATILITY: {'min': 30, 'max': 90, 'optimal': 60},
                MarketRegime.LOW_VOLATILITY: {'min': 90, 'max': 180, 'optimal': 120},
                MarketRegime.BREAKOUT: {'min': 15, 'max': 45, 'optimal': 30},
                MarketRegime.CRISIS: {'min': 8, 'max': 20, 'optimal': 15},
                MarketRegime.NORMAL: {'min': 45, 'max': 120, 'optimal': 75}
            },
            'aggressive': {
                MarketRegime.BULL_TREND: {'min': 60, 'max': 120, 'optimal': 80},
                MarketRegime.BEAR_TREND: {'min': 40, 'max': 80, 'optimal': 60},
                MarketRegime.SIDEWAYS: {'min': 15, 'max': 40, 'optimal': 25},
                MarketRegime.HIGH_VOLATILITY: {'min': 20, 'max': 60, 'optimal': 40},
                MarketRegime.LOW_VOLATILITY: {'min': 60, 'max': 120, 'optimal': 80},
                MarketRegime.BREAKOUT: {'min': 10, 'max': 30, 'optimal': 20},
                MarketRegime.CRISIS: {'min': 5, 'max': 15, 'optimal': 10},
                MarketRegime.NORMAL: {'min': 30, 'max': 80, 'optimal': 50}
            }
        }
    
    def _initialize_default_configs(self):
        """初始化默认配置"""
        # 使用平衡配置作为默认
        config_level = 'balanced'
        
        for regime, config in self.base_config[config_level].items():
            window_config = WindowConfig(
                name=f"{regime.value}_{config_level}",
                min_length=config['min'],
                max_length=config['max'],
                optimal_length=config['optimal'],
                performance_score=0.5,
                market_regime=regime,
                last_updated=datetime.now(),
                usage_count=0
            )
            self.window_configs[regime].append(window_config)
            self.current_windows[regime] = window_config
    
    def get_optimal_window(self, 
                          market_state: MarketState,
                          data_length: int) -> WindowConfig:
        """
        获取最优时间窗口
        
        Args:
            market_state: 当前市场状态
            data_length: 可用数据长度
            
        Returns:
            WindowConfig: 最优窗口配置
        """
        try:
            regime = market_state.regime
            available_configs = self.window_configs[regime]
            
            if not available_configs:
                # 如果没有配置，创建一个默认配置
                default_config = self._create_default_config(regime, data_length)
                available_configs.append(default_config)
                self.current_windows[regime] = default_config
                return default_config
            
            # 过滤适合数据长度的配置
            suitable_configs = [
                config for config in available_configs 
                if config.min_length <= data_length <= config.max_length
            ]
            
            if not suitable_configs:
                # 如果没有完全合适的配置，选择最接近的
                suitable_configs = sorted(
                    available_configs,
                    key=lambda x: abs(x.optimal_length - data_length)
                )[:1]
            
            # 基于性能和置信度选择最佳配置
            best_config = self._select_best_config(suitable_configs, market_state)
            
            # 更新使用统计
            best_config.usage_count += 1
            self.current_windows[regime] = best_config
            
            return best_config
            
        except Exception as e:
            self.logger.error(f"获取最优窗口失败: {e}")
            # 返回一个保守的默认配置
            return self._create_default_config(MarketRegime.NORMAL, data_length)
    
    def _create_default_config(self, regime: MarketRegime, data_length: int) -> WindowConfig:
        """创建默认配置"""
        # 基础窗口长度基于数据长度和市场状态
        if data_length >= 252:
            base_length = min(120, data_length // 2)
        elif data_length >= 60:
            base_length = min(60, data_length // 2)
        else:
            base_length = max(20, data_length // 2)
        
        # 根据市场状态调整
        if regime in [MarketRegime.CRISIS, MarketRegime.BREAKOUT]:
            min_length = max(5, base_length // 4)
            max_length = max(min_length + 5, base_length // 2)
        elif regime in [MarketRegime.SIDEWAYS, MarketRegime.LOW_VOLATILITY]:
            min_length = max(20, base_length // 2)
            max_length = max(min_length + 10, base_length)
        else:
            min_length = max(30, base_length // 2)
            max_length = max(min_length + 20, base_length)
        
        return WindowConfig(
            name=f"default_{regime.value}",
            min_length=min_length,
            max_length=max_length,
            optimal_length=base_length,
            performance_score=0.5,
            market_regime=regime,
            last_updated=datetime.now(),
            usage_count=0
        )
    
    def _select_best_config(self, 
                           configs: List[WindowConfig], 
                           market_state: MarketState) -> WindowConfig:
        """选择最佳配置"""
        if len(configs) == 1:
            return configs[0]
        
        # 评分标准：性能分数、市场状态匹配度、置信度
        scored_configs = []
        
        for config in configs:
            # 基础性能分数
            performance_score = config.performance_score
            
            # 置信度调整
            confidence_bonus = market_state.confidence * 0.1
            
            # 趋势强度匹配度
            trend_match = 1 - abs(config.optimal_length - 60) / 100  # 假设60是中性长度
            trend_match = max(0, trend_match)
            
            # 波动率匹配度
            vol_bonus = 0
            if market_state.volatility_level > 0.7 and config.optimal_length < 50:  # 高波动对应短窗口
                vol_bonus = 0.1
            elif market_state.volatility_level < 0.3 and config.optimal_length > 100:  # 低波动对应长窗口
                vol_bonus = 0.1
            
            final_score = performance_score + confidence_bonus + trend_match * 0.2 + vol_bonus
            scored_configs.append((config, final_score))
        
        # 返回评分最高的配置
        best_config = max(scored_configs, key=lambda x: x[1])[0]
        return best_config
    
    def update_performance(self, 
                          window_config: WindowConfig, 
                          performance_metrics: Dict[str, float]):
        """更新窗口性能"""
        try:
            # 更新配置性能分数
            mse = performance_metrics.get('mse', 0.0)
            mae = performance_metrics.get('mae', 0.0)
            
            # 将MSE/MAE转换为性能分数 [0, 1]
            # 假设合理的MSE范围是0.1-1.0
            performance_score = max(0, 1 - mse)
            
            # 使用指数移动平均更新性能分数
            alpha = 0.3
            window_config.performance_score = (
                alpha * performance_score + 
                (1 - alpha) * window_config.performance_score
            )
            
            # 记录历史性能
            performance_record = {
                'window_config': window_config.name,
                'market_regime': window_config.market_regime.value,
                'performance_score': performance_score,
                'timestamp': datetime.now(),
                'metrics': performance_metrics
            }
            self.historical_performance.append(performance_record)
            
            # 更新最后更新时间
            window_config.last_updated = datetime.now()
            
        except Exception as e:
            self.logger.error(f"更新窗口性能失败: {e}")
    
    def adapt_windows(self, market_state: MarketState):
        """自适应调整窗口配置"""
        try:
            regime = market_state.regime
            current_config = self.current_windows.get(regime)
            
            if not current_config:
                return
            
            # 检查是否需要调整
            if market_state.confidence < 0.5 or market_state.volatility_level > 0.8:
                # 高不确定性或极高波动率时采用保守策略
                new_length = min(current_config.optimal_length + 20, current_config.max_length)
            elif market_state.confidence > 0.8 and market_state.volatility_level < 0.3:
                # 低波动率且高置信度时采用积极策略
                new_length = max(current_config.optimal_length - 10, current_config.min_length)
            else:
                return  # 不需要调整
            
            # 创建新配置
            new_config = WindowConfig(
                name=f"{current_config.name}_adapted_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                min_length=current_config.min_length,
                max_length=current_config.max_length,
                optimal_length=new_length,
                performance_score=current_config.performance_score,
                market_regime=regime,
                last_updated=datetime.now(),
                usage_count=0
            )
            
            # 添加到配置列表
            self.window_configs[regime].append(new_config)
            self.current_windows[regime] = new_config
            
            self.logger.info(f"窗口配置已调整: {regime.value} -> {new_length}")
            
        except Exception as e:
            self.logger.error(f"自适应调整失败: {e}")
    
    def get_window_history(self) -> List[Dict[str, Any]]:
        """获取窗口历史记录"""
        return [record for record in self.historical_performance]
    
    def export_config(self, filepath: str):
        """导出配置到文件"""
        try:
            export_data = {
                'window_configs': {
                    regime.value: [config.to_dict() for config in configs]
                    for regime, configs in self.window_configs.items()
                },
                'current_windows': {
                    regime.value: config.to_dict()
                    for regime, config in self.current_windows.items()
                },
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"配置已导出到: {filepath}")
            
        except Exception as e:
            self.logger.error(f"导出配置失败: {e}")
    
    def import_config(self, filepath: str):
        """从文件导入配置"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # 恢复窗口配置
            for regime_str, configs_data in import_data.get('window_configs', {}).items():
                regime = MarketRegime(regime_str)
                configs = []
                for config_data in configs_data:
                    config = WindowConfig(
                        name=config_data['name'],
                        min_length=config_data['min_length'],
                        max_length=config_data['max_length'],
                        optimal_length=config_data['optimal_length'],
                        performance_score=config_data['performance_score'],
                        market_regime=regime,
                        last_updated=datetime.fromisoformat(config_data['last_updated']),
                        usage_count=config_data['usage_count']
                    )
                    configs.append(config)
                self.window_configs[regime] = configs
            
            # 恢复当前窗口
            for regime_str, config_data in import_data.get('current_windows', {}).items():
                regime = MarketRegime(regime_str)
                # 找到对应的配置对象
                for config in self.window_configs.get(regime, []):
                    if config.name == config_data['name']:
                        self.current_windows[regime] = config
                        break
            
            self.logger.info(f"配置已从 {filepath} 导入")
            
        except Exception as e:
            self.logger.error(f"导入配置失败: {e}")


class WindowOptimizer:
    """窗口优化器"""
    
    def __init__(self, 
                 optimization_window: int = 30,
                 min_improvement: float = 0.01):
        """
        初始化窗口优化器
        
        Args:
            optimization_window: 优化评估窗口
            min_improvement: 最小改进阈值
        """
        self.optimization_window = optimization_window
        self.min_improvement = min_improvement
        self.optimization_history: deque = deque(maxlen=100)
        self.logger = logging.getLogger(__name__)
    
    def optimize_window_config(self, 
                              window_manager: AdaptiveWindowManager,
                              market_state: MarketState,
                              historical_data: pd.DataFrame,
                              target_variable: str = 'market_forward_excess_returns',
                              feature_columns: List[str] = None) -> WindowConfig:
        """
        优化窗口配置
        
        Args:
            window_manager: 窗口管理器
            market_state: 当前市场状态
            historical_data: 历史数据
            target_variable: 目标变量名
            feature_columns: 特征列名
            
        Returns:
            WindowConfig: 优化后的窗口配置
        """
        try:
            current_config = window_manager.get_optimal_window(
                market_state, len(historical_data)
            )
            
            # 评估不同窗口长度的性能
            candidate_lengths = self._generate_candidate_lengths(
                current_config, len(historical_data)
            )
            
            performance_results = []
            
            for length in candidate_lengths:
                performance = self._evaluate_window_performance(
                    historical_data, length, target_variable, feature_columns
                )
                performance_results.append((length, performance))
            
            # 选择最佳窗口长度
            best_length, best_performance = max(performance_results, key=lambda x: x[1])
            
            # 检查是否比当前配置有显著改进
            current_performance = self._evaluate_window_performance(
                historical_data, current_config.optimal_length, target_variable, feature_columns
            )
            
            improvement = (best_performance - current_performance) / current_performance
            
            if improvement > self.min_improvement:
                # 创建优化后的配置
                optimized_config = WindowConfig(
                    name=f"optimized_{current_config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    min_length=current_config.min_length,
                    max_length=current_config.max_length,
                    optimal_length=best_length,
                    performance_score=best_performance,
                    market_regime=market_state.regime,
                    last_updated=datetime.now(),
                    usage_count=0
                )
                
                # 记录优化结果
                optimization_record = {
                    'original_config': current_config.name,
                    'optimized_config': optimized_config.name,
                    'original_length': current_config.optimal_length,
                    'optimized_length': best_length,
                    'improvement': improvement,
                    'market_regime': market_state.regime.value,
                    'timestamp': datetime.now(),
                    'performance': best_performance
                }
                self.optimization_history.append(optimization_record)
                
                self.logger.info(f"窗口配置已优化: {current_config.optimal_length} -> {best_length} "
                               f"(改进: {improvement:.3f})")
                
                return optimized_config
            else:
                self.logger.info(f"优化结果无显著改进，使用当前配置")
                return current_config
                
        except Exception as e:
            self.logger.error(f"窗口优化失败: {e}")
            return window_manager.get_optimal_window(market_state, len(historical_data))
    
    def _generate_candidate_lengths(self, 
                                  current_config: WindowConfig, 
                                  data_length: int) -> List[int]:
        """生成候选窗口长度"""
        # 在当前配置的范围内生成候选长度
        candidates = set()
        
        # 添加当前配置的长度
        candidates.add(current_config.optimal_length)
        
        # 添加配置边界
        candidates.add(current_config.min_length)
        candidates.add(current_config.max_length)
        
        # 添加中间值
        for i in range(1, 5):
            intermediate = current_config.min_length + (
                (current_config.max_length - current_config.min_length) * i // 5
            )
            candidates.add(intermediate)
        
        # 过滤有效长度
        valid_candidates = [
            length for length in candidates
            if current_config.min_length <= length <= current_config.max_length
            and length <= data_length
        ]
        
        return sorted(valid_candidates)[:7]  # 最多返回7个候选
    
    def _evaluate_window_performance(self, 
                                   data: pd.DataFrame, 
                                   window_length: int,
                                   target_variable: str,
                                   feature_columns: List[str] = None) -> float:
        """
        评估特定窗口长度的性能
        
        Args:
            data: 数据
            window_length: 窗口长度
            target_variable: 目标变量
            feature_columns: 特征列
            
        Returns:
            float: 性能分数
        """
        try:
            if window_length >= len(data):
                return 0.0
            
            # 选择窗口数据
            window_data = data.tail(window_length).copy()
            
            if feature_columns is None:
                # 自动检测特征列
                feature_columns = [col for col in data.columns if col.startswith(('D', 'E', 'I', 'M', 'P', 'S', 'V', 'MOM'))]
            
            if not feature_columns or target_variable not in data.columns:
                return 0.0
            
            # 准备数据
            X = window_data[feature_columns].fillna(0)
            y = window_data[target_variable].fillna(0)
            
            if len(X) < 10 or len(y) < 10:
                return 0.0
            
            # 简单的线性回归评估
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import cross_val_score
            
            model = LinearRegression()
            scores = cross_val_score(model, X, y, cv=min(3, len(X)//5), scoring='neg_mean_squared_error')
            
            # 转换为正的性能分数
            performance = -np.mean(scores)
            performance = max(0, 1 - performance)  # 转换为[0,1]范围
            
            return performance
            
        except Exception as e:
            self.logger.warning(f"窗口性能评估失败 (length={window_length}): {e}")
            return 0.0
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """获取优化历史"""
        return [record for record in self.optimization_history]


class PerformanceTracker:
    """性能跟踪器"""
    
    def __init__(self, 
                 tracking_window: int = 100,
                 alert_threshold: float = 0.1):
        """
        初始化性能跟踪器
        
        Args:
            tracking_window: 跟踪窗口
            alert_threshold: 告警阈值
        """
        self.tracking_window = tracking_window
        self.alert_threshold = alert_threshold
        
        # 性能数据存储
        self.performance_records: deque = deque(maxlen=tracking_window)
        self.regime_performance: Dict[MarketRegime, List[float]] = defaultdict(list)
        self.window_performance: Dict[str, List[float]] = defaultdict(list)
        
        # 告警系统
        self.alerts: List[Dict[str, Any]] = []
        self.performance_baselines: Dict[str, float] = {}
        
        self.logger = logging.getLogger(__name__)
    
    def record_performance(self, 
                          window_config: WindowConfig,
                          market_state: MarketState,
                          performance_metrics: Dict[str, float],
                          model_predictions: np.ndarray = None,
                          actual_values: np.ndarray = None):
        """
        记录性能数据
        
        Args:
            window_config: 使用的窗口配置
            market_state: 当前市场状态
            performance_metrics: 性能指标
            model_predictions: 模型预测值
            actual_values: 实际值
        """
        try:
            # 创建性能记录
            record = {
                'timestamp': datetime.now(),
                'window_config': window_config.name,
                'market_regime': market_state.regime.value,
                'window_length': window_config.optimal_length,
                'performance_metrics': performance_metrics.copy(),
                'market_confidence': market_state.confidence,
                'volatility_level': market_state.volatility_level
            }
            
            # 添加预测准确性指标
            if model_predictions is not None and actual_values is not None:
                record['prediction_metrics'] = {
                    'mse': mean_squared_error(actual_values, model_predictions),
                    'mae': mean_absolute_error(actual_values, model_predictions),
                    'r2': self._calculate_r2(actual_values, model_predictions)
                }
            
            self.performance_records.append(record)
            
            # 更新市场状态性能统计
            mse = performance_metrics.get('mse', 0.0)
            self.regime_performance[market_state.regime].append(mse)
            
            # 更新窗口性能统计
            self.window_performance[window_config.name].append(mse)
            
            # 检查是否需要告警
            self._check_alerts(window_config, market_state, performance_metrics)
            
        except Exception as e:
            self.logger.error(f"记录性能数据失败: {e}")
    
    def _calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算R²分数"""
        try:
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        except:
            return 0.0
    
    def _check_alerts(self, 
                     window_config: WindowConfig,
                     market_state: MarketState,
                     performance_metrics: Dict[str, float]):
        """检查性能告警"""
        try:
            mse = performance_metrics.get('mse', 0.0)
            
            # 性能下降告警
            recent_performance = self.get_recent_performance(10)
            if len(recent_performance) >= 5:
                current_avg = np.mean([r['performance_metrics'].get('mse', 0) for r in recent_performance[-5:]])
                baseline_avg = np.mean([r['performance_metrics'].get('mse', 0) for r in recent_performance[:-5]])
                
                if current_avg > baseline_avg * (1 + self.alert_threshold):
                    self._create_alert(
                        'PERFORMANCE_DEGRADATION',
                        f"性能下降: {current_avg:.4f} > {baseline_avg:.4f} * {(1 + self.alert_threshold):.2f}",
                        {'current': current_avg, 'baseline': baseline_avg, 'threshold': self.alert_threshold}
                    )
            
            # 高波动率低性能告警
            if market_state.volatility_level > 0.8 and mse > 0.5:
                self._create_alert(
                    'HIGH_VOL_LOW_PERFORMANCE',
                    f"高波动率({market_state.volatility_level:.2f})下性能差(MSE={mse:.4f})",
                    {'volatility': market_state.volatility_level, 'mse': mse}
                )
            
            # 窗口长度不匹配告警
            regime_recent = [r for r in recent_performance 
                           if r['market_regime'] == market_state.regime.value]
            if regime_recent:
                avg_length = np.mean([r['window_length'] for r in regime_recent[-5:]])
                if abs(window_config.optimal_length - avg_length) > 30:
                    self._create_alert(
                        'WINDOW_MISMATCH',
                        f"窗口长度异常: 当前={window_config.optimal_length}, 平均={avg_length:.0f}",
                        {'current': window_config.optimal_length, 'average': avg_length}
                    )
            
        except Exception as e:
            self.logger.error(f"检查告警失败: {e}")
    
    def _create_alert(self, alert_type: str, message: str, details: Dict[str, Any]):
        """创建告警"""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now(),
            'details': details
        }
        self.alerts.append(alert)
        
        # 保持最近的100个告警
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        
        self.logger.warning(f"性能告警 [{alert_type}]: {message}")
    
    def get_performance_summary(self, 
                               regime: Optional[MarketRegime] = None) -> Dict[str, Any]:
        """获取性能摘要"""
        try:
            # 获取相关记录
            if regime:
                records = [r for r in self.performance_records 
                          if r['market_regime'] == regime.value]
            else:
                records = list(self.performance_records)
            
            if not records:
                return {'error': 'No performance data available'}
            
            # 计算统计指标
            mse_values = [r['performance_metrics'].get('mse', 0) for r in records]
            mae_values = [r['performance_metrics'].get('mae', 0) for r in records]
            
            summary = {
                'total_records': len(records),
                'mse': {
                    'mean': np.mean(mse_values),
                    'std': np.std(mse_values),
                    'min': np.min(mse_values),
                    'max': np.max(mse_values)
                },
                'mae': {
                    'mean': np.mean(mae_values),
                    'std': np.std(mae_values),
                    'min': np.min(mae_values),
                    'max': np.max(mae_values)
                }
            }
            
            # 添加市场状态分布
            regime_counts = {}
            for record in records:
                regime_name = record['market_regime']
                regime_counts[regime_name] = regime_counts.get(regime_name, 0) + 1
            summary['regime_distribution'] = regime_counts
            
            return summary
            
        except Exception as e:
            self.logger.error(f"获取性能摘要失败: {e}")
            return {'error': str(e)}
    
    def get_recent_performance(self, n: int = 10) -> List[Dict[str, Any]]:
        """获取最近的性能记录"""
        return list(self.performance_records)[-n:]
    
    def get_alerts(self, alert_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取告警"""
        if alert_type:
            return [alert for alert in self.alerts if alert['type'] == alert_type]
        return list(self.alerts)
    
    def export_performance_report(self, filepath: str):
        """导出性能报告"""
        try:
            report = {
                'summary': self.get_performance_summary(),
                'recent_performance': self.get_recent_performance(20),
                'alerts': self.get_alerts(),
                'regime_performance': {
                    regime.value: {
                        'count': len(performances),
                        'mean_mse': np.mean(performances) if performances else 0,
                        'std_mse': np.std(performances) if performances else 0
                    }
                    for regime, performances in self.regime_performance.items()
                },
                'window_performance': {
                    window_name: {
                        'count': len(performances),
                        'mean_mse': np.mean(performances) if performances else 0
                    }
                    for window_name, performances in self.window_performance.items()
                },
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"性能报告已导出到: {filepath}")
            
        except Exception as e:
            self.logger.error(f"导出性能报告失败: {e}")


# 自适应时间窗口系统主类
class AdaptiveTimeWindowSystem:
    """自适应时间窗口系统主类"""
    
    def __init__(self, 
                 config_level: str = 'balanced',
                 lookback_periods: int = 252,
                 tracking_enabled: bool = True):
        """
        初始化自适应时间窗口系统
        
        Args:
            config_level: 配置级别 ('conservative', 'balanced', 'aggressive')
            lookback_periods: 回看期间
            tracking_enabled: 是否启用性能跟踪
        """
        # 初始化各个组件
        self.detector = MarketStateDetector(lookback_periods=lookback_periods)
        self.window_manager = AdaptiveWindowManager()
        self.optimizer = WindowOptimizer()
        
        self.tracking_enabled = tracking_enabled
        self.tracker = PerformanceTracker() if tracking_enabled else None
        
        self.config_level = config_level
        self.logger = logging.getLogger(__name__)
        
        # 系统状态
        self.is_initialized = False
        self.last_market_state = None
        self.current_window_config = None
        
    def initialize(self, historical_data: pd.DataFrame, 
                  price_col: str = 'P1', 
                  volume_col: str = None):
        """初始化系统"""
        try:
            self.logger.info("初始化自适应时间窗口系统...")
            
            # 检测初始市场状态
            self.last_market_state = self.detector.detect_market_state(
                historical_data, price_col, volume_col
            )
            
            # 获取初始窗口配置
            self.current_window_config = self.window_manager.get_optimal_window(
                self.last_market_state, len(historical_data)
            )
            
            self.is_initialized = True
            self.logger.info(f"系统初始化完成。当前市场状态: {self.last_market_state.regime.value}, "
                           f"推荐窗口: {self.current_window_config.optimal_length}")
            
        except Exception as e:
            self.logger.error(f"系统初始化失败: {e}")
            raise
    
    def get_optimal_window(self, 
                          new_data: pd.DataFrame,
                          price_col: str = 'P1',
                          volume_col: str = None) -> WindowConfig:
        """获取最优窗口配置"""
        if not self.is_initialized:
            raise RuntimeError("系统未初始化，请先调用initialize()方法")
        
        try:
            # 检测当前市场状态
            current_state = self.detector.detect_market_state(
                new_data, price_col, volume_col
            )
            
            # 检查市场状态是否发生显著变化
            if (self.last_market_state is None or 
                self._is_significant_state_change(self.last_market_state, current_state)):
                
                self.logger.info(f"市场状态变化: {self.last_market_state.regime.value} -> {current_state.regime.value}")
                
                # 自适应调整窗口
                self.window_manager.adapt_windows(current_state)
                
                # 优化窗口配置
                optimized_config = self.optimizer.optimize_window_config(
                    self.window_manager, current_state, new_data
                )
                
                # 更新当前配置
                if optimized_config.name != self.current_window_config.name:
                    self.current_window_config = optimized_config
                    self.logger.info(f"窗口配置已更新: {optimized_config.optimal_length}")
            
            else:
                # 使用现有配置或重新获取最优配置
                self.current_window_config = self.window_manager.get_optimal_window(
                    current_state, len(new_data)
                )
            
            # 更新状态
            self.last_market_state = current_state
            
            return self.current_window_config
            
        except Exception as e:
            self.logger.error(f"获取最优窗口失败: {e}")
            # 返回默认配置
            return self.window_manager.get_optimal_window(
                MarketState(
                    regime=MarketRegime.NORMAL,
                    trend_strength=0.0,
                    volatility_level=0.5,
                    volume_anomaly=0.0,
                    confidence=0.5,
                    timestamp=datetime.now()
                ),
                len(new_data)
            )
    
    def _is_significant_state_change(self, 
                                   old_state: MarketState, 
                                   new_state: MarketState) -> bool:
        """判断市场状态是否发生显著变化"""
        # 市场状态类型变化
        if old_state.regime != new_state.regime:
            return True
        
        # 置信度显著变化
        if abs(old_state.confidence - new_state.confidence) > 0.3:
            return True
        
        # 波动率显著变化
        if abs(old_state.volatility_level - new_state.volatility_level) > 0.4:
            return True
        
        # 趋势强度显著变化
        if abs(old_state.trend_strength - new_state.trend_strength) > 0.5:
            return True
        
        return False
    
    def update_performance(self, performance_metrics: Dict[str, float]):
        """更新性能数据"""
        if not self.tracking_enabled or self.tracker is None:
            return
        
        try:
            self.tracker.record_performance(
                self.current_window_config,
                self.last_market_state,
                performance_metrics
            )
        except Exception as e:
            self.logger.error(f"更新性能数据失败: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'is_initialized': self.is_initialized,
            'config_level': self.config_level,
            'tracking_enabled': self.tracking_enabled,
            'last_market_state': self.last_market_state.to_dict() if self.last_market_state else None,
            'current_window_config': self.current_window_config.to_dict() if self.current_window_config else None,
            'performance_summary': self.tracker.get_performance_summary() if self.tracker else None,
            'recent_alerts': self.tracker.get_alerts()[-5:] if self.tracker else []
        }
    
    def export_system_state(self, filepath: str):
        """导出系统状态"""
        try:
            # 导出窗口管理器配置
            window_manager_path = filepath.replace('.json', '_windows.json')
            self.window_manager.export_config(window_manager_path)
            
            # 导出性能跟踪器报告
            if self.tracker:
                tracker_path = filepath.replace('.json', '_performance.json')
                self.tracker.export_performance_report(tracker_path)
            
            # 导出系统状态
            system_state = self.get_system_status()
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(system_state, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"系统状态已导出到: {filepath}")
            
        except Exception as e:
            self.logger.error(f"导出系统状态失败: {e}")