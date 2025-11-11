"""
时间序列验证系统与现有Hull Tactical系统的集成模块

集成功能:
- 与现有模型系统(HullModel)的无缝集成
- 与特征工程系统(FeaturePipeline)的配合
- 与自适应时间窗口系统的协调
- 模型集成策略的验证增强
- 配置系统的统一管理
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from .time_series_validation import (
    TimeSeriesCrossValidator, ValidationConfig, ValidationResult, 
    ValidationStrategy, FinanceValidationMetrics, MarketRegime,
    create_time_series_validator
)

# 现有系统集成
from .lib.models import HullModel, DynamicWeightedEnsemble, StackingEnsemble
from .lib.features import FeaturePipeline, get_feature_columns
from .lib.config import get_config

logger = logging.getLogger(__name__)


class IntegratedTimeSeriesValidator:
    """集成时间序列验证器 - 与Hull Tactical系统深度集成"""
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """初始化集成验证器
        
        Args:
            config_override: 配置覆盖选项
        """
        # 加载现有配置
        try:
            hull_config = get_config()
            validation_config = self._create_enhanced_config(hull_config, config_override)
        except Exception as e:
            logger.warning(f"无法加载Hull配置，使用默认配置: {e}")
            validation_config = ValidationConfig()
        
        self.validator = TimeSeriesCrossValidator(validation_config)
        self.hull_config = hull_config if 'hull_config' in locals() else None
        self.feature_pipeline = None
        self.model_registry = {}
        self.validation_history = []
        
        # 集成专用设置
        self.auto_feature_engineering = True
        self.auto_model_selection = True
        self.enable_adaptive_splits = True
        self.performance_baseline = None
        
    def _create_enhanced_config(self, hull_config, override: Optional[Dict]) -> ValidationConfig:
        """基于Hull配置创建增强验证配置"""
        try:
            # 从Hull配置中提取相关参数
            features_config = hull_config.get_features_config()
            model_config = hull_config.get_model_config()
            
            # 基础验证配置
            base_config = ValidationConfig(
                strategy=ValidationStrategy.EXPANDING_WINDOW,  # 使用扩展窗口作为默认
                n_splits=5,
                test_size=min(252, max(50, int(features_config.get('test_size', 126)))),
                min_train_samples=features_config.get('min_train_samples', 500),
                enable_performance_monitoring=True,
                enable_market_regime_detection=True,
                verbose=True
            )
            
            # 应用覆盖参数
            if override:
                for key, value in override.items():
                    if hasattr(base_config, key):
                        setattr(base_config, key, value)
            
            return base_config
            
        except Exception as e:
            logger.warning(f"配置创建失败，使用默认配置: {e}")
            return ValidationConfig()
    
    def validate_hull_model(self, 
                           model_type: str = "lightgbm",
                           model_params: Optional[Dict] = None,
                           ensemble_config: Optional[Dict] = None,
                           train_data: Optional[pd.DataFrame] = None,
                           target_column: str = "forward_returns",
                           feature_columns: Optional[List[str]] = None,
                           additional_validation_configs: Optional[List[ValidationConfig]] = None) -> Dict[str, ValidationResult]:
        """验证Hull模型系统 - 全面验证"""
        
        # 1. 加载或准备数据
        if train_data is None:
            raise ValueError("需要提供训练数据")
        
        # 2. 特征工程
        if self.auto_feature_engineering:
            features = self._prepare_features(train_data, feature_columns)
        else:
            feature_cols = feature_columns or get_feature_columns(train_data)
            features = train_data[feature_cols]
        
        # 3. 准备目标变量
        if target_column not in train_data.columns:
            raise ValueError(f"目标列 {target_column} 不存在于数据中")
        
        target = train_data[target_column]
        
        # 4. 准备额外数据（市场数据等）
        additional_data = self._prepare_additional_data(train_data, features)
        
        # 5. 创建Hull模型
        hull_model = HullModel(
            model_type=model_type,
            model_params=model_params or {},
            ensemble_config=ensemble_config or {}
        )
        
        # 6. 执行多策略验证
        validation_results = {}
        
        # 主要验证策略
        primary_strategies = [
            ValidationStrategy.EXPANDING_WINDOW,
            ValidationStrategy.PURGED_TIME_SERIES,
            ValidationStrategy.MARKET_REGIME_BASED
        ]
        
        for strategy in primary_strategies:
            try:
                config = ValidationConfig(strategy=strategy, verbose=True)
                validator = TimeSeriesCrossValidator(config)
                result = validator.validate(hull_model, features, target, additional_data)
                validation_results[strategy.value] = result
                
                logger.info(f"{strategy.value} 验证完成")
                
            except Exception as e:
                logger.error(f"{strategy.value} 验证失败: {e}")
                continue
        
        # 7. 执行额外配置验证
        if additional_validation_configs:
            for i, config in enumerate(additional_validation_configs):
                try:
                    validator = TimeSeriesCrossValidator(config)
                    result = validator.validate(hull_model, features, target, additional_data)
                    validation_results[f"custom_config_{i}"] = result
                    
                except Exception as e:
                    logger.error(f"自定义配置 {i} 验证失败: {e}")
                    continue
        
        # 8. 存储验证历史
        self.validation_history.append({
            'timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'n_strategies': len(validation_results),
            'total_samples': len(features),
            'feature_count': features.shape[1]
        })
        
        return validation_results
    
    def validate_ensemble_performance(self,
                                    base_models: List[str] = None,
                                    ensemble_configs: List[Dict] = None,
                                    train_data: Optional[pd.DataFrame] = None,
                                    target_column: str = "forward_returns") -> Dict[str, Dict[str, ValidationResult]]:
        """验证集成模型性能 - 多种集成策略比较"""
        
        if train_data is None:
            raise ValueError("需要提供训练数据")
        
        if base_models is None:
            base_models = ['lightgbm', 'xgboost', 'catboost']
        
        if ensemble_configs is None:
            ensemble_configs = [
                {'type': 'dynamic_weighted', 'config': {'performance_window': 50}},
                {'type': 'stacking', 'config': {'cv_folds': 3}},
                {'type': 'averaging', 'config': {'weights': [0.4, 0.3, 0.3]}}
            ]
        
        ensemble_results = {}
        
        for config in ensemble_configs:
            ensemble_type = config['type']
            ensemble_config = config['config']
            
            logger.info(f"验证集成策略: {ensemble_type}")
            
            try:
                # 创建集成模型配置
                if ensemble_type == 'dynamic_weighted':
                    final_model_type = "dynamic_weighted_ensemble"
                elif ensemble_type == 'stacking':
                    final_model_type = "stacking_ensemble"
                elif ensemble_type == 'averaging':
                    final_model_type = "ensemble"
                else:
                    final_model_type = ensemble_type
                
                # 执行验证
                results = self.validate_hull_model(
                    model_type=final_model_type,
                    model_params={'weights': ensemble_config.get('weights')},
                    ensemble_config=ensemble_config,
                    train_data=train_data,
                    target_column=target_column
                )
                
                ensemble_results[ensemble_type] = results
                
            except Exception as e:
                logger.error(f"集成策略 {ensemble_type} 验证失败: {e}")
                continue
        
        return ensemble_results
    
    def adaptive_validation_sequence(self,
                                   train_data: pd.DataFrame,
                                   target_column: str = "forward_returns",
                                   max_rounds: int = 3) -> List[Dict[str, Any]]:
        """自适应验证序列 - 逐步优化验证策略"""
        
        validation_sequence = []
        
        for round_num in range(max_rounds):
            logger.info(f"执行自适应验证轮次: {round_num + 1}/{max_rounds}")
            
            # 1. 根据数据和历史表现选择策略
            if round_num == 0:
                # 第一轮：保守策略
                strategies = [ValidationStrategy.TIME_SERIES_SPLIT]
                configs = [ValidationConfig(n_splits=3, verbose=True)]
            elif round_num == max_rounds - 1:
                # 最后一轮：全面策略
                strategies = [ValidationStrategy.EXPANDING_WINDOW, ValidationStrategy.PURGED_K_FOLD]
                configs = [
                    ValidationConfig(n_splits=7, enable_nested_cv=True),
                    ValidationConfig(n_splits=5, enable_purged_cv=True)
                ]
            else:
                # 中间轮次：平衡策略
                strategies = [ValidationStrategy.MARKET_REGIME_BASED, ValidationStrategy.VOLATILITY_TIERED]
                configs = [
                    ValidationConfig(enable_performance_monitoring=True),
                    ValidationConfig(enable_market_regime_detection=True)
                ]
            
            # 2. 执行策略验证
            round_results = {}
            for strategy, config in zip(strategies, configs):
                try:
                    # 准备数据
                    feature_cols = get_feature_columns(train_data)
                    features = train_data[feature_cols]
                    target = train_data[target_column]
                    additional_data = self._prepare_additional_data(train_data, features)
                    
                    # 创建简单模型进行验证
                    model = HullModel(model_type="lightgbm")
                    
                    # 验证
                    validator = TimeSeriesCrossValidator(config)
                    result = validator.validate(model, features, target, additional_data)
                    round_results[strategy.value] = result
                    
                except Exception as e:
                    logger.error(f"轮次 {round_num} 策略 {strategy.value} 失败: {e}")
                    continue
            
            # 3. 评估结果并选择最佳策略
            best_strategy = None
            best_score = float('-inf')
            
            for strategy_name, result in round_results.items():
                if 'mse' in result.metrics:
                    # 对于MSE，越小越好，所以取负值
                    avg_mse = np.mean(result.metrics['mse'])
                    score = -avg_mse
                    
                    if score > best_score:
                        best_score = score
                        best_strategy = strategy_name
            
            # 4. 记录轮次结果
            round_summary = {
                'round_number': round_num,
                'strategies_tested': list(round_results.keys()),
                'best_strategy': best_strategy,
                'best_score': best_score,
                'results': {k: v.get_summary() for k, v in round_results.items()},
                'timestamp': datetime.now().isoformat()
            }
            
            validation_sequence.append(round_summary)
            
            # 5. 根据结果调整下一轮配置
            if best_strategy and round_num < max_rounds - 1:
                # 基于最佳策略优化下一轮配置
                logger.info(f"最佳策略: {best_strategy}, 分数: {best_score:.6f}")
        
        return validation_sequence
    
    def compare_with_baseline(self,
                            model: HullModel,
                            baseline_model: HullModel,
                            X: pd.DataFrame,
                            y: pd.Series,
                            additional_data: Optional[Dict] = None) -> Dict[str, ValidationResult]:
        """与基线模型比较验证"""
        
        results = {}
        
        # 验证当前模型
        current_result = self.validator.validate(model, X, y, additional_data)
        results['current_model'] = current_result
        
        # 验证基线模型
        baseline_result = self.validator.validate(baseline_model, X, y, additional_data)
        results['baseline_model'] = baseline_result
        
        # 计算改进指标
        improvements = self._calculate_improvements(current_result, baseline_result)
        results['improvements'] = improvements
        
        return results
    
    def _prepare_features(self, train_data: pd.DataFrame, 
                         feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """准备特征 - 集成特征工程"""
        
        if self.feature_pipeline is None:
            # 创建增强特征工程管道
            self.feature_pipeline = FeaturePipeline(
                enable_intelligent_selection=True,
                enable_feature_combinations=True,
                enable_tiered_features=True,
                max_features=200,  # 限制特征数量
                stateful=True
            )
            
            # 获取特征列
            if feature_columns is None:
                feature_columns = get_feature_columns(train_data)
            
            # 拟合并转换
            features = self.feature_pipeline.fit_transform(train_data, feature_columns)
        else:
            # 转换新数据
            features = self.feature_pipeline.transform(train_data)
        
        logger.info(f"特征工程完成: {features.shape}")
        return features
    
    def _prepare_additional_data(self, train_data: pd.DataFrame, 
                               features: pd.DataFrame) -> Dict[str, Any]:
        """准备额外数据用于高级验证"""
        
        additional_data = {}
        
        # 市场数据
        if 'P1' in train_data.columns:
            prices = train_data['P1'].values
            additional_data['prices'] = prices
            
            # 计算收益和波动率
            if len(prices) > 1:
                returns = np.diff(prices) / prices[:-1]
                additional_data['returns'] = returns
                additional_data['volatility'] = np.abs(returns)
            
            # 市场状态检测
            if 'V1' in train_data.columns:
                additional_data['market_volatility'] = train_data['V1'].values
        
        # 基准和市场收益
        if 'forward_returns' in train_data.columns:
            additional_data['benchmark_returns'] = train_data['forward_returns'].values
        
        # 滞后数据
        lagged_cols = [col for col in train_data.columns if col.startswith('lagged_')]
        if lagged_cols:
            additional_data['lagged_data'] = train_data[lagged_cols].to_dict('list')
        
        # 市场状态标识
        if len(features) > 50:  # 需要足够的数据进行状态检测
            if 'V1' in features.columns:
                volatility = features['V1'].values
                market_regimes = self._detect_market_regimes_from_features(volatility)
                additional_data['regimes'] = market_regimes
        
        return additional_data
    
    def _detect_market_regimes_from_features(self, volatility: np.ndarray) -> List[str]:
        """从特征中检测市场状态"""
        
        regimes = []
        window = 20
        
        for i in range(len(volatility)):
            if i < window:
                regimes.append("unknown")
                continue
            
            # 计算波动率状态
            recent_vol = np.std(volatility[i-window:i])
            historical_vol = np.std(volatility[:i]) if i > window else recent_vol
            vol_ratio = recent_vol / (historical_vol + 1e-8)
            
            # 状态分类
            if vol_ratio > 1.5:
                regime = "high_volatility"
            elif vol_ratio < 0.7:
                regime = "low_volatility"
            else:
                regime = "normal"
            
            regimes.append(regime)
        
        return regimes
    
    def _calculate_improvements(self, current_result: ValidationResult, 
                               baseline_result: ValidationResult) -> Dict[str, float]:
        """计算改进指标"""
        
        improvements = {}
        
        for metric in current_result.metrics:
            if metric in baseline_result.metrics:
                current_mean = np.mean(current_result.metrics[metric])
                baseline_mean = np.mean(baseline_result.metrics[metric])
                
                if metric in ['mse', 'mae']:  # 越小越好的指标
                    improvement = (baseline_mean - current_mean) / baseline_mean
                else:  # 越大越好的指标
                    improvement = (current_mean - baseline_mean) / baseline_mean
                
                improvements[f'{metric}_improvement'] = improvement
                improvements[f'{metric}_absolute'] = current_mean - baseline_mean
        
        return improvements
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """获取验证摘要"""
        
        if not self.validation_history:
            return {"message": "无验证历史"}
        
        summary = {
            'total_validations': len(self.validation_history),
            'validation_types': set(entry['model_type'] for entry in self.validation_history),
            'latest_validation': self.validation_history[-1],
            'average_samples': np.mean([entry['total_samples'] for entry in self.validation_history]),
            'average_features': np.mean([entry['feature_count'] for entry in self.validation_history])
        }
        
        return summary
    
    def save_validation_results(self, results: Dict[str, ValidationResult], 
                               output_dir: Path, prefix: str = "validation"):
        """保存验证结果"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for strategy_name, result in results.items():
            filename = f"{prefix}_{strategy_name}_{timestamp}"
            
            # 保存结果
            result_path = output_dir / f"{filename}.json"
            summary_path = output_dir / f"{filename}_summary.json"
            
            try:
                # 保存完整结果
                self.validator.save_results(result, result_path)
                
                # 保存摘要
                summary = result.get_summary()
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2, default=str)
                
                logger.info(f"验证结果已保存: {filename}")
                
            except Exception as e:
                logger.error(f"保存结果失败 {strategy_name}: {e}")


class TimeSeriesValidationAPI:
    """时间序列验证API - 简化的用户接口"""
    
    @staticmethod
    def quick_validate(model: HullModel, 
                      train_data: pd.DataFrame,
                      target_column: str = "forward_returns",
                      strategy: str = "expanding_window") -> Dict[str, Any]:
        """快速验证接口"""
        
        validator = IntegratedTimeSeriesValidator()
        
        results = validator.validate_hull_model(
            model_type=model.model_type,
            model_params=model.model_params,
            ensemble_config=model.ensemble_config,
            train_data=train_data,
            target_column=target_column
        )
        
        # 返回摘要
        return {
            strategy: results[strategy].get_summary() if strategy in results else {}
        }
    
    @staticmethod
    def comprehensive_validation(train_data: pd.DataFrame,
                               model_types: List[str] = None,
                               strategies: List[str] = None,
                               target_column: str = "forward_returns") -> Dict[str, Any]:
        """全面验证接口"""
        
        if model_types is None:
            model_types = ['lightgbm', 'xgboost', 'dynamic_weighted_ensemble']
        
        if strategies is None:
            strategies = ['expanding_window', 'purged_time_series', 'market_regime_based']
        
        validator = IntegratedTimeSeriesValidator()
        comprehensive_results = {}
        
        for model_type in model_types:
            model_results = {}
            for strategy in strategies:
                try:
                    config = ValidationConfig(strategy=ValidationStrategy(strategy))
                    integrated_validator = TimeSeriesCrossValidator(config)
                    
                    feature_cols = get_feature_columns(train_data)
                    features = train_data[feature_cols]
                    target = train_data[target_column]
                    
                    model = HullModel(model_type=model_type)
                    result = integrated_validator.validate(model, features, target)
                    
                    model_results[strategy] = result.get_summary()
                    
                except Exception as e:
                    model_results[strategy] = {"error": str(e)}
            
            comprehensive_results[model_type] = model_results
        
        return comprehensive_results


# 便利函数
def validate_with_time_series_cv(model: HullModel, 
                                train_data: pd.DataFrame,
                                strategy: str = "expanding_window",
                                **kwargs) -> ValidationResult:
    """便利函数：使用时间序列交叉验证验证模型"""
    
    api = TimeSeriesValidationAPI()
    result = api.quick_validate(model, train_data, strategy=strategy, **kwargs)
    
    # 返回ValidationResult对象
    strategy_result = result.get(strategy, {})
    if "timestamp" in strategy_result:
        return ValidationResult(
            strategy=ValidationStrategy(strategy),
            n_splits=strategy_result.get("n_splits", 1),
            metrics=strategy_result.get("metrics", {})
        )
    else:
        raise ValueError(f"验证失败: {strategy_result}")


def comprehensive_model_validation(train_data: pd.DataFrame,
                                 model_types: List[str] = None,
                                 strategies: List[str] = None,
                                 save_results: bool = True,
                                 output_dir: str = "validation_results") -> Dict[str, Any]:
    """全面模型验证函数"""
    
    api = TimeSeriesValidationAPI()
    results = api.comprehensive_validation(
        train_data=train_data,
        model_types=model_types,
        strategies=strategies
    )
    
    if save_results:
        validator = IntegratedTimeSeriesValidator()
        output_path = Path(output_dir)
        
        for model_type, model_results in results.items():
            for strategy, result_summary in model_results.items():
                if "error" not in result_summary:
                    filename = f"{model_type}_{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    # 这里可以保存详细的JSON结果
                    logger.info(f"结果可保存到: {output_path / filename}")
    
    return results


# 导出
__all__ = [
    'IntegratedTimeSeriesValidator',
    'TimeSeriesValidationAPI', 
    'validate_with_time_series_cv',
    'comprehensive_model_validation'
]