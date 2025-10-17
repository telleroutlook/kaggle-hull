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
        
        self.config['features'] = {
            'max_features': '20',
            'rolling_windows': '[5, 10, 20]',
            'lag_periods': '[1, 2, 3]'
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
            'lag_periods': eval(self.config.get('features', 'lag_periods', fallback='[1, 2, 3]'))
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


# 全局配置实例
config_manager = ConfigManager()


def get_config() -> ConfigManager:
    """获取全局配置管理器实例"""
    return config_manager


__all__ = [
    "ConfigManager",
    "get_config"
]