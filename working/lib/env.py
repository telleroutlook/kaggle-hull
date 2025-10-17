"""
环境配置和路径管理工具
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DataPaths:
    """数据路径容器"""
    
    train_data: Path
    test_data: Path
    kaggle_evaluation: Path


@dataclass(frozen=True)
class LogPaths:
    """日志路径容器"""
    
    log_jsonl: Path
    metrics_csv: Path
    submission_file: Path


def detect_run_environment() -> str:
    """检测当前运行环境"""
    
    cwd = Path.cwd()
    if "/kaggle/" in cwd.as_posix():
        return "kaggle"
    if (PROJECT_ROOT / "input").exists():
        return "local"
    return "unknown"


def get_data_paths(env: str) -> DataPaths:
    """根据环境返回数据路径"""
    
    if env == "kaggle":
        base = Path("/kaggle/input/hull-tactical-market-prediction")
    else:
        base = PROJECT_ROOT / "input" / "hull-tactical-market-prediction"
    
    return DataPaths(
        train_data=base / "train.csv",
        test_data=base / "test.csv", 
        kaggle_evaluation=base / "kaggle_evaluation"
    )


def get_log_paths(env: str) -> LogPaths:
    """根据环境返回日志路径"""
    
    if env == "kaggle":
        base = Path("/kaggle/working")
    else:
        base = PROJECT_ROOT / "working"
    
    return LogPaths(
        log_jsonl=base / "hull_logs.jsonl",
        metrics_csv=base / "hull_metrics.csv",
        submission_file=base / "submission.parquet"
    )


__all__ = [
    "DataPaths",
    "LogPaths", 
    "PROJECT_ROOT",
    "detect_run_environment",
    "get_data_paths",
    "get_log_paths",
]