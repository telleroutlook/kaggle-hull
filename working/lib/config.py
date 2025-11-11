"""
配置管理模块
"""

from __future__ import annotations

import configparser
import os
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = configparser.ConfigParser()
        
        # 如果没有指定配置文件路径，使用默认路径
        if config_path is None:
            # 尝试在当前目录查找配置文件
            default_paths = [
                "config.ini",
                "working/config.ini",
                "../working/config.ini"
            ]
            
            for path in default_paths:
                if os.path.exists(path):
                    config_path = path
                    break
            else:
                # 如果找不到配置文件，使用默认配置
                self._set_default_config()
                return
        
        # 读取配置文件
        if os.path.exists(config_path):
            self.config.read(config_path)
        else:
            print(f"⚠️ 配置文件 {config_path} 不存在，使用默认配置")
            self._set_default_config()
    
    def _set_default_config(self):
        """设置默认配置"""
        self.config['model'] = {
            'type': 'baseline',
            'baseline_n_estimators': '100',
            'baseline_max_depth': '10',
            'baseline_random_state': '42'
        }
        
        # 超参数调优结果存储
        self.config['tuned_parameters'] = {
            'lightgbm_params': '{}',
            'xgboost_params': '{}',
            'catboost_params': '{}',
            'random_forest_params': '{}',
            'last_tuning_date': '',
            'best_model_type': 'lightgbm',
            'tuning_enabled': 'False'
        }
        
        # 调优配置
        self.config['tuning'] = {
            'n_trials': '50',
            'cv_folds': '5',
            'search_strategy': 'optuna',
            'validation_strategy': 'time_series',
            'primary_metric': 'mse',
            'timeout_seconds': '1800'
        }
        
        self.config['features'] = {
            'max_features': '20',
            'rolling_windows': '[5, 10, 20]',
            'lag_periods': '[1, 2, 3]',
            'enable_data_quality': 'True',
            'enable_feature_stability': 'True',
            'outlier_detection': 'True',
            'missing_value_strategy': 'median',
            'enable_intelligent_selection': 'True',
            'enable_feature_combinations': 'True',
            'enable_tiered_features': 'True',
            'enable_robust_scaling': 'True',
            'feature_selection_method': 'mixed',
            'combination_complexity': '3',
            'tiered_levels': '4'
        }
        
        self.config['data'] = {
            'train_test_split_ratio': '0.8',
            'validation_split_ratio': '0.1'
        }
        
        self.config['evaluation'] = {
            'volatility_constraint': '1.2',
            'risk_free_rate': '0.0'
        }
        
        self.config['logging'] = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
        
        # 自适应时间窗口配置
        self.config['adaptive_windows'] = {
            'enabled': 'True',
            'config_level': 'balanced',  # conservative, balanced, aggressive
            'lookback_periods': '252',
            'volatility_window': '20',
            'trend_windows': '[5, 10, 20, 50]',
            'volume_window': '20',
            'optimization_window': '30',
            'min_improvement': '0.01',
            'tracking_enabled': 'True',
            'tracking_window': '100',
            'alert_threshold': '0.1',
            'memory_size': '1000',
            'adaptation_threshold': '0.1',
            'max_window_length': '252',
            'min_window_length': '10'
        }
        
        # 窗口配置预设
        self.config['window_presets'] = {
            'conservative_bull_trend': 'min:120,max:252,optimal:180',
            'conservative_bear_trend': 'min:90,max:180,optimal:120',
            'conservative_sideways': 'min:30,max:90,optimal:60',
            'conservative_high_vol': 'min:45,max:120,optimal:75',
            'conservative_low_vol': 'min:120,max:252,optimal:180',
            'conservative_breakout': 'min:20,max:60,optimal:40',
            'conservative_crisis': 'min:10,max:30,optimal:20',
            'conservative_normal': 'min:60,max:180,optimal:120',
            
            'balanced_bull_trend': 'min:90,max:180,optimal:120',
            'balanced_bear_trend': 'min:60,max:120,optimal:90',
            'balanced_sideways': 'min:20,max:60,optimal:40',
            'balanced_high_vol': 'min:30,max:90,optimal:60',
            'balanced_low_vol': 'min:90,max:180,optimal:120',
            'balanced_breakout': 'min:15,max:45,optimal:30',
            'balanced_crisis': 'min:8,max:20,optimal:15',
            'balanced_normal': 'min:45,max:120,optimal:75',
            
            'aggressive_bull_trend': 'min:60,max:120,optimal:80',
            'aggressive_bear_trend': 'min:40,max:80,optimal:60',
            'aggressive_sideways': 'min:15,max:40,optimal:25',
            'aggressive_high_vol': 'min:20,max:60,optimal:40',
            'aggressive_low_vol': 'min:60,max:120,optimal:80',
            'aggressive_breakout': 'min:10,max:30,optimal:20',
            'aggressive_crisis': 'min:5,max:15,optimal:10',
            'aggressive_normal': 'min:30,max:80,optimal:50'
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return {
            'type': self.config.get('model', 'type', fallback='baseline'),
            'baseline_n_estimators': self.config.getint('model', 'baseline_n_estimators', fallback=100),
            'baseline_max_depth': self.config.getint('model', 'baseline_max_depth', fallback=10),
            'baseline_random_state': self.config.getint('model', 'baseline_random_state', fallback=42)
        }
    
    def get_features_config(self) -> Dict[str, Any]:
        """获取特征工程配置"""
        return {
            'max_features': self.config.getint('features', 'max_features', fallback=20),
            'rolling_windows': eval(self.config.get('features', 'rolling_windows', fallback='[5, 10, 20]')),
            'lag_periods': eval(self.config.get('features', 'lag_periods', fallback='[1, 2, 3]')),
            'enable_data_quality': self.config.getboolean('features', 'enable_data_quality', fallback=True),
            'enable_feature_stability': self.config.getboolean('features', 'enable_feature_stability', fallback=True),
            'outlier_detection': self.config.getboolean('features', 'outlier_detection', fallback=True),
            'missing_value_strategy': self.config.get('features', 'missing_value_strategy', fallback='median'),
            'enable_intelligent_selection': self.config.getboolean('features', 'enable_intelligent_selection', fallback=True),
            'enable_feature_combinations': self.config.getboolean('features', 'enable_feature_combinations', fallback=True),
            'enable_tiered_features': self.config.getboolean('features', 'enable_tiered_features', fallback=True),
            'enable_robust_scaling': self.config.getboolean('features', 'enable_robust_scaling', fallback=True),
            'feature_selection_method': self.config.get('features', 'feature_selection_method', fallback='mixed'),
            'combination_complexity': self.config.getint('features', 'combination_complexity', fallback=3),
            'tiered_levels': self.config.getint('features', 'tiered_levels', fallback=4)
        }
    
    def get_data_config(self) -> Dict[str, Any]:
        """获取数据配置"""
        return {
            'train_test_split_ratio': self.config.getfloat('data', 'train_test_split_ratio', fallback=0.8),
            'validation_split_ratio': self.config.getfloat('data', 'validation_split_ratio', fallback=0.1)
        }
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """获取评估配置"""
        return {
            'volatility_constraint': self.config.getfloat('evaluation', 'volatility_constraint', fallback=1.2),
            'risk_free_rate': self.config.getfloat('evaluation', 'risk_free_rate', fallback=0.0)
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """获取日志配置"""
        return {
            'level': self.config.get('logging', 'level', fallback='INFO'),
            'format': self.config.get('logging', 'format', fallback='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        }
    
    def get_adaptive_windows_config(self) -> Dict[str, Any]:
        """获取自适应时间窗口配置"""
        return {
            'enabled': self.config.getboolean('adaptive_windows', 'enabled', fallback=True),
            'config_level': self.config.get('adaptive_windows', 'config_level', fallback='balanced'),
            'lookback_periods': self.config.getint('adaptive_windows', 'lookback_periods', fallback=252),
            'volatility_window': self.config.getint('adaptive_windows', 'volatility_window', fallback=20),
            'trend_windows': eval(self.config.get('adaptive_windows', 'trend_windows', fallback='[5, 10, 20, 50]')),
            'volume_window': self.config.getint('adaptive_windows', 'volume_window', fallback=20),
            'optimization_window': self.config.getint('adaptive_windows', 'optimization_window', fallback=30),
            'min_improvement': self.config.getfloat('adaptive_windows', 'min_improvement', fallback=0.01),
            'tracking_enabled': self.config.getboolean('adaptive_windows', 'tracking_enabled', fallback=True),
            'tracking_window': self.config.getint('adaptive_windows', 'tracking_window', fallback=100),
            'alert_threshold': self.config.getfloat('adaptive_windows', 'alert_threshold', fallback=0.1),
            'memory_size': self.config.getint('adaptive_windows', 'memory_size', fallback=1000),
            'adaptation_threshold': self.config.getfloat('adaptive_windows', 'adaptation_threshold', fallback=0.1),
            'max_window_length': self.config.getint('adaptive_windows', 'max_window_length', fallback=252),
            'min_window_length': self.config.getint('adaptive_windows', 'min_window_length', fallback=10)
        }
    
    def get_window_preset(self, level: str, regime: str) -> Dict[str, int]:
        """
        获取窗口预设配置
        
        Args:
            level: 配置级别 (conservative, balanced, aggressive)
            regime: 市场状态 (bull_trend, bear_trend, sideways, high_vol, low_vol, breakout, crisis, normal)
            
        Returns:
            Dict[str, int]: 包含min_length, max_length, optimal_length的字典
        """
        preset_key = f"{level}_{regime}"
        preset_str = self.config.get('window_presets', preset_key, fallback='')
        
        if not preset_str:
            return {'min_length': 30, 'max_length': 120, 'optimal_length': 60}
        
        try:
            # 解析 "min:120,max:252,optimal:180" 格式
            parts = preset_str.split(',')
            result = {}
            for part in parts:
                key, value = part.split(':')
                result[f"{key}_length"] = int(value)
            return result
        except (ValueError, TypeError):
            return {'min_length': 30, 'max_length': 120, 'optimal_length': 60}
    
    def update_adaptive_windows_config(self, **kwargs):
        """更新自适应时间窗口配置"""
        valid_keys = {
            'enabled': 'bool',
            'config_level': 'str',
            'lookback_periods': 'int',
            'volatility_window': 'int',
            'trend_windows': 'list',
            'volume_window': 'int',
            'optimization_window': 'int',
            'min_improvement': 'float',
            'tracking_enabled': 'bool',
            'tracking_window': 'int',
            'alert_threshold': 'float',
            'memory_size': 'int',
            'adaptation_threshold': 'float',
            'max_window_length': 'int',
            'min_window_length': 'int'
        }
        
        for key, value in kwargs.items():
            if key in valid_keys:
                if valid_keys[key] == 'bool':
                    self.config.set('adaptive_windows', key, str(bool(value)))
                elif valid_keys[key] == 'int':
                    self.config.set('adaptive_windows', key, str(int(value)))
                elif valid_keys[key] == 'float':
                    self.config.set('adaptive_windows', key, str(float(value)))
                elif valid_keys[key] == 'list':
                    self.config.set('adaptive_windows', key, str(value))
                else:
                    self.config.set('adaptive_windows', key, str(value))
    
    # 超参数调优相关方法
    def get_tuned_parameters(self, model_type: str) -> Dict[str, Any]:
        """获取调优后的参数"""
        param_key = f"{model_type.lower()}_params"
        param_str = self.config.get('tuned_parameters', param_key, fallback='{}')
        
        try:
            import ast
            return ast.literal_eval(param_str)
        except (ValueError, SyntaxError):
            return {}
    
    def save_tuned_parameters(self, model_type: str, params: Dict[str, Any], 
                            best_score: float = None, tuning_date: str = None):
        """保存调优后的参数"""
        param_key = f"{model_type.lower()}_params"
        
        # 保存参数
        import json
        param_str = json.dumps(params, ensure_ascii=False)
        self.config.set('tuned_parameters', param_key, param_str)
        
        # 更新调优日期
        if tuning_date is None:
            from datetime import datetime
            tuning_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.config.set('tuned_parameters', 'last_tuning_date', tuning_date)
        
        # 如果提供了分数，更新最佳模型类型
        if best_score is not None:
            current_best = self.get_best_model_type()
            if not current_best or model_type.lower() == current_best.lower():
                self.config.set('tuned_parameters', 'best_model_type', model_type.lower())
        
        self.config.set('tuned_parameters', 'tuning_enabled', 'True')
    
    def get_best_model_type(self) -> str:
        """获取最佳模型类型"""
        return self.config.get('tuned_parameters', 'best_model_type', fallback='lightgbm')
    
    def is_tuning_enabled(self) -> bool:
        """检查是否启用了调优"""
        return self.config.getboolean('tuned_parameters', 'tuning_enabled', fallback=False)
    
    def get_tuning_config(self) -> Dict[str, Any]:
        """获取调优配置"""
        return {
            'n_trials': self.config.getint('tuning', 'n_trials', fallback=50),
            'cv_folds': self.config.getint('tuning', 'cv_folds', fallback=5),
            'search_strategy': self.config.get('tuning', 'search_strategy', fallback='optuna'),
            'validation_strategy': self.config.get('tuning', 'validation_strategy', fallback='time_series'),
            'primary_metric': self.config.get('tuning', 'primary_metric', fallback='mse'),
            'timeout_seconds': self.config.getint('tuning', 'timeout_seconds', fallback=1800)
        }
    
    def update_tuning_config(self, **kwargs):
        """更新调优配置"""
        for key, value in kwargs.items():
            if key in ['n_trials', 'cv_folds', 'timeout_seconds']:
                self.config.set('tuning', key, str(int(value)))
            else:
                self.config.set('tuning', key, str(value))
    
    def get_optimized_model_params(self, model_type: str) -> Dict[str, Any]:
        """获取优化后的模型参数（如果没有调优则使用默认参数）"""
        tuned_params = self.get_tuned_parameters(model_type)
        
        if tuned_params:
            return tuned_params
        else:
            # 返回默认参数
            default_params = {
                'lightgbm': {
                    'n_estimators': 2000,
                    'learning_rate': 0.015,
                    'num_leaves': 256,
                    'max_depth': 10,
                    'subsample': 0.9,
                    'colsample_bytree': 0.9,
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.8,
                    'min_child_samples': 8
                },
                'xgboost': {
                    'n_estimators': 3000,
                    'learning_rate': 0.01,
                    'max_depth': 7,
                    'subsample': 0.8,
                    'colsample_bytree': 0.7,
                    'reg_lambda': 2.0,
                    'reg_alpha': 0.0,
                    'gamma': 0.0,
                    'min_child_weight': 10
                },
                'catboost': {
                    'iterations': 5000,
                    'learning_rate': 0.02,
                    'depth': 7,
                    'l2_leaf_reg': 4.0,
                    'bagging_temperature': 0.8,
                    'random_strength': 0.8
                }
            }
            
            return default_params.get(model_type.lower(), {})


# 全局配置实例
config_manager = ConfigManager()


def get_config() -> ConfigManager:
    """获取全局配置管理器实例"""
    return config_manager


__all__ = [
    "ConfigManager",
    "get_config"
]