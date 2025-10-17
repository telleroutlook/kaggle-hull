"""
测试工具函数
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from lib.utils import (
        PerformanceTracker, 
        save_logs, 
        save_metrics, 
        validate_submission,
        print_progress
    )
except ImportError:
    # 如果lib.utils导入失败，尝试直接导入
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
    from utils import (
        PerformanceTracker, 
        save_logs, 
        save_metrics, 
        validate_submission,
        print_progress
    )


def test_performance_tracker():
    """测试性能跟踪器"""
    
    tracker = PerformanceTracker()
    
    # 测试任务计时
    tracker.start_task("test_task")
    import time
    time.sleep(0.1)  # 等待一小段时间
    tracker.end_task()
    
    # 测试指标记录
    tracker.log_metric("accuracy", 0.95)
    tracker.log_metric("loss", 0.05)
    
    # 获取摘要
    summary = tracker.get_summary()
    
    # 验证摘要结构
    assert 'total_time_seconds' in summary
    assert 'task_breakdown' in summary
    assert 'metrics' in summary
    assert 'test_task' in summary['task_breakdown']
    assert summary['metrics']['accuracy'] == 0.95
    assert summary['metrics']['loss'] == 0.05


def test_save_logs():
    """测试日志保存"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        log_path = Path(f.name)
    
    try:
        logs = {
            'task1_time': 1.5,
            'task2_time': 2.3,
            'status': 'success'
        }
        
        save_logs(logs, log_path)
        
        # 验证文件创建
        assert log_path.exists()
        
        # 验证内容
        with open(log_path, 'r') as f:
            content = f.read()
            assert 'task1_time' in content
            assert 'task2_time' in content
            assert 'timestamp' in content
            
    finally:
        log_path.unlink()  # 清理临时文件


def test_save_metrics():
    """测试指标保存"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        metrics_path = Path(f.name)
    
    try:
        metrics1 = {
            'accuracy': 0.95,
            'loss': 0.05,
            'epoch': 1
        }
        
        metrics2 = {
            'accuracy': 0.96,
            'loss': 0.04,
            'epoch': 2
        }
        
        # 第一次保存
        save_metrics(metrics1, metrics_path)
        
        # 第二次保存（追加）
        save_metrics(metrics2, metrics_path)
        
        # 验证文件内容和结构
        df = pd.read_csv(metrics_path)
        assert len(df) == 2
        assert 'accuracy' in df.columns
        assert 'loss' in df.columns
        assert 'timestamp' in df.columns
        assert df['accuracy'].iloc[0] == 0.95
        assert df['accuracy'].iloc[1] == 0.96
        
    finally:
        metrics_path.unlink()  # 清理临时文件


def test_validate_submission():
    """测试提交文件验证"""
    
    # 测试有效提交
    valid_df = pd.DataFrame({
        'date_id': [1, 2, 3],
        'prediction': [0.5, 1.2, 0.8]
    })
    assert validate_submission(valid_df) == True
    
    # 测试缺失列
    missing_col_df = pd.DataFrame({
        'date_id': [1, 2, 3]
    })
    assert validate_submission(missing_col_df) == False
    
    # 测试超出范围的预测值
    out_of_range_df = pd.DataFrame({
        'date_id': [1, 2, 3],
        'prediction': [-0.1, 1.5, 2.5]
    })
    assert validate_submission(out_of_range_df) == False
    
    # 测试包含缺失值
    missing_values_df = pd.DataFrame({
        'date_id': [1, 2, 3],
        'prediction': [0.5, np.nan, 0.8]
    })
    assert validate_submission(missing_values_df) == False


def test_print_progress():
    """测试进度条打印（主要验证没有异常）"""
    
    # 这个函数主要输出到控制台，我们主要验证它不会抛出异常
    try:
        print_progress(50, 100, "Testing: ")
        print_progress(100, 100, "Testing: ")
        print("\n✅ 进度条测试完成")
        success = True
    except Exception as e:
        success = False
        print(f"进度条测试失败: {e}")
    
    assert success == True


if __name__ == "__main__":
    test_performance_tracker()
    test_save_logs()
    test_save_metrics()
    test_validate_submission()
    test_print_progress()
    print("✅ 所有工具函数测试通过")