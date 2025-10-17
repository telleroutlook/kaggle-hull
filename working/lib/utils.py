"""
通用工具函数
"""

from __future__ import annotations

import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class PerformanceTracker:
    """性能跟踪器"""
    
    def __init__(self):
        self.start_time = time.time()
        self.task_times = []
        self.metrics = {}
    
    def start_task(self, task_name: str):
        """开始任务计时"""
        self.current_task = task_name
        self.task_start_time = time.time()
    
    def end_task(self):
        """结束任务计时"""
        if hasattr(self, 'current_task') and hasattr(self, 'task_start_time'):
            duration = time.time() - self.task_start_time
            self.task_times.append((self.current_task, duration))
    
    def log_metric(self, name: str, value: Any):
        """记录指标"""
        self.metrics[name] = value
    
    def get_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        total_time = time.time() - self.start_time
        
        summary = {
            'total_time_seconds': total_time,
            'total_time_minutes': total_time / 60,
            'task_breakdown': dict(self.task_times),
            'metrics': self.metrics
        }
        
        return summary


def save_logs(logs: Dict[str, Any], log_path: Path):
    """保存日志到JSONL文件"""
    
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        **logs
    }
    
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry) + '\n')


def save_metrics(metrics: Dict[str, float], metrics_path: Path):
    """保存指标到CSV文件"""
    
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df['timestamp'] = datetime.now().isoformat()
    
    if metrics_path.exists():
        existing_df = pd.read_csv(metrics_path)
        metrics_df = pd.concat([existing_df, metrics_df], ignore_index=True)
    
    metrics_df.to_csv(metrics_path, index=False)


def print_progress(current: int, total: int, prefix: str = "", suffix: str = ""):
    """打印进度条"""
    
    bar_length = 50
    progress = float(current) / total
    arrow = "=" * int(round(progress * bar_length) - 1) + ">"
    spaces = " " * (bar_length - len(arrow))
    
    print(f'\r{prefix}[{arrow + spaces}] {int(progress * 100)}% {suffix}', end='', flush=True)
    
    if current == total:
        print()


def validate_submission(submission_df: pd.DataFrame) -> bool:
    """验证提交文件格式"""
    
    required_columns = ['date_id', 'prediction']
    
    # 检查必要列
    missing_cols = [col for col in required_columns if col not in submission_df.columns]
    if missing_cols:
        print(f"错误: 提交文件缺少列: {missing_cols}")
        return False
    
    # 检查预测值范围
    predictions = submission_df['prediction'].values
    if np.any(predictions < 0) or np.any(predictions > 2):
        print(f"错误: 预测值必须在0-2之间, 当前范围: [{np.min(predictions):.4f}, {np.max(predictions):.4f}]")
        return False
    
    # 检查缺失值
    if submission_df.isnull().any().any():
        print("错误: 提交文件包含缺失值")
        return False
    
    print("✅ 提交文件验证通过")
    return True


__all__ = [
    "PerformanceTracker",
    "save_logs", 
    "save_metrics",
    "print_progress",
    "validate_submission",
]