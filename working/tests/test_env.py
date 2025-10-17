"""
测试环境检测和路径管理
"""

import sys
import os
import pytest
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from lib.env import detect_run_environment, get_data_paths, DataPaths
except ImportError:
    # 如果lib.env导入失败，尝试直接导入
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
    from env import detect_run_environment, get_data_paths, DataPaths


def test_detect_run_environment():
    """测试环境检测功能"""
    
    # 测试本地环境检测
    env = detect_run_environment()
    assert env in ["kaggle", "local", "unknown"]
    

def test_get_data_paths():
    """测试路径获取功能"""
    
    # 测试kaggle环境
    kaggle_paths = get_data_paths("kaggle")
    assert isinstance(kaggle_paths, DataPaths)
    assert "/kaggle/input/hull-tactical-market-prediction/" in str(kaggle_paths.train_data)
    
    # 测试本地环境
    local_paths = get_data_paths("local")
    assert isinstance(local_paths, DataPaths)
    assert "input/hull-tactical-market-prediction/" in str(local_paths.train_data)
    

def test_data_paths_dataclass():
    """测试DataPaths数据类"""
    
    paths = DataPaths(
        train_data=Path("train.csv"),
        test_data=Path("test.csv"),
        evaluation_dir=Path("kaggle_evaluation")
    )
    
    assert paths.train_data == Path("train.csv")
    assert paths.test_data == Path("test.csv")
    assert paths.evaluation_dir == Path("kaggle_evaluation")


if __name__ == "__main__":
    test_detect_run_environment()
    test_get_data_paths() 
    test_data_paths_dataclass()
    print("✅ 所有环境测试通过")