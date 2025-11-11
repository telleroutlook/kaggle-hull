"""警告处理工具模块"""

import threading
import warnings
from typing import Set

# 延迟导入pandas以避免循环依赖
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# 线程安全的警告配置跟踪
_configured_modules: Set[str] = set()
_config_lock = threading.Lock()

# 备用全局变量（用于向后兼容）
_warnings_configured = False


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
    if HAS_PANDAS:
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
    """线程安全的单例警告配置"""
    module_name = __name__
    
    with _config_lock:
        if module_name in _configured_modules:
            return
        
        # 执行实际的警告配置
        setup_warnings_handling()
        _configured_modules.add(module_name)
    
    # 保持向后兼容性
    global _warnings_configured
    _warnings_configured = True


def is_warnings_configured() -> bool:
    """检查警告是否已配置"""
    with _config_lock:
        return len(_configured_modules) > 0 or _warnings_configured


def reset_warnings_config():
    """重置警告配置状态（主要用于测试）"""
    with _config_lock:
        _configured_modules.clear()
    global _warnings_configured
    _warnings_configured = False