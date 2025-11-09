"""警告处理工具模块"""

import warnings
import numpy as np
import pandas as pd


def setup_warnings_handling():
    """设置警告处理策略"""
    
    # 配置pandas警告
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='pandas')
    warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
    
    # 配置numpy警告  
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
    
    # 配置scipy警告
    try:
        warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy')
    except:
        pass
    
    # 忽略特定的比较警告（这些警告在NaN处理后应该是无害的）
    warnings.filterwarnings('ignore', 
                          message='.*invalid value encountered in greater.*',
                          category=RuntimeWarning)
    warnings.filterwarnings('ignore',
                          message='.*invalid value encountered in less.*', 
                          category=RuntimeWarning)
    
    # 忽略调试器冻结模块警告
    warnings.filterwarnings('ignore',
                          message='.*frozen modules.*',
                          category=UserWarning)
    
    # 设置pandas选项以更好地处理NaN
    pd.set_option('mode.chained_assignment', None)  # 忽略链式赋值警告
    pd.set_option('display.max_columns', None)      # 避免列显示警告
    
    print("✅ 警告处理已配置")


def setup_warnings_early():
    """在早期阶段设置警告处理（在pandas导入之前）"""
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # 特别处理pandas比较警告
    warnings.filterwarnings('ignore', 
                          message='.*invalid value encountered in greater.*',
                          category=RuntimeWarning)
    warnings.filterwarnings('ignore',
                          message='.*invalid value encountered in less.*', 
                          category=RuntimeWarning)
    warnings.filterwarnings('ignore',
                          message='.*frozen modules.*',
                          category=UserWarning)


def suppress_specific_warnings():
    """临时抑制特定警告的上下文管理器"""
    
    class WarningSuppressor:
        def __enter__(self):
            # 保存当前警告过滤器
            self._original_filters = warnings.filters[:]
            
            # 添加抑制规则
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            warnings.filterwarnings('ignore', category=FutureWarning)
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            # 恢复原始警告过滤器
            warnings.filters[:] = self._original_filters
    
    return WarningSuppressor()


# 全局警告处理实例
_warnings_configured = False

def ensure_warnings_configured():
    """确保警告处理已配置（调用一次即可）"""
    global _warnings_configured
    if not _warnings_configured:
        setup_warnings_handling()
        _warnings_configured = True