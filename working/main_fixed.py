#!/usr/bin/env python3
"""
Hull Tactical - Market Prediction 主模型文件
Kaggle竞赛的模型入口点
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加lib目录到路径
working_dir = os.path.dirname(__file__)
if working_dir:
    sys.path.insert(0, working_dir)
else:
    # 如果__file__为空（在某些环境中），使用当前工作目录
    sys.path.insert(0, os.getcwd())

import logging

from lib.env import detect_run_environment, get_data_paths, get_log_paths
from lib.data import load_test_data, validate_data
from lib.features import engineer_features, get_feature_columns
from lib.models import HullModel, create_submission
from lib.utils import PerformanceTracker, save_logs, save_metrics, validate_submission, write_result_json

# 尝试导入配置模块
try:
    from lib.config import ConfigManager, get_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("⚠️ 配置模块不可用")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args(argv=None):
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Hull Tactical - Market Prediction 模型")
    
    parser.add_argument(
        "--model-type",
        choices=["baseline", "lightgbm", "xgboost", "ensemble"],
        default="baseline",
        help="选择模型类型 (默认: baseline)"
    )
    
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="显式指定数据路径（覆盖自动检测）"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出文件路径（默认：根据环境自动设置）"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="配置文件路径"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出模式"
    )
    
    # 处理不同的运行环境参数
    if argv is None:
        # 过滤掉Jupyter内核参数和不相关的参数
        filtered_argv = [arg for arg in sys.argv[1:] if not arg.startswith('-f') and not ':memory:' in arg]
        try:
            return parser.parse_args(filtered_argv)
        except SystemExit:
            # 如果解析失败，使用默认参数
            print("⚠️ 参数解析失败，使用默认参数")
            return parser.parse_args([])
    
    return parser.parse_args(argv)


def main():
    """主函数 - 运行模型预测"""
    
    args = parse_args()
    
    # 初始化配置
    if CONFIG_AVAILABLE and args.config:
        config_manager = ConfigManager(str(args.config))
        logging_config = config_manager.get_logging_config()
        # 更新日志配置
        logging.basicConfig(
            level=getattr(logging, logging_config['level'], logging.INFO),
            format=logging_config['format']
        )
    elif CONFIG_AVAILABLE:
        config_manager = get_config()
    else:
        config_manager = None
    
    tracker = PerformanceTracker(logger=logger)
    
    print("🚀 Hull Tactical - Market Prediction 模型启动")
    print(f"📋 模型类型: {args.model_type}")
    
    # 检测运行环境
    env = detect_run_environment()
    data_paths = get_data_paths(env)
    log_paths = get_log_paths(env)
    result_path = Path("/kaggle/working/result.json") if env == "kaggle" else log_paths.log_jsonl.parent / "result.json"
    
    print(f"🏠 运行环境: {env}")
    print(f"📁 数据路径: {data_paths.test_data}")
    
    # 设置输出路径
    if args.output:
        submission_path = args.output
    else:
        submission_path = log_paths.submission_file
    
    try:
        # 加载数据
        tracker.start_task("load_data")
        tracker.record_memory_usage()
        test_data = load_test_data(data_paths)
        
        if not validate_data(test_data, "test"):
            return 1
        
        tracker.record_memory_usage()
        tracker.end_task()
        
        # 特征工程
        tracker.start_task("feature_engineering")
        tracker.record_memory_usage()
        feature_cols = get_feature_columns(test_data)
        features = engineer_features(test_data, feature_cols)
        
        if args.verbose:
            print(f"🔧 特征数量: {len(feature_cols)}")
            print(f"📊 特征形状: {features.shape}")
        
        tracker.record_memory_usage()
        tracker.end_task()
        
        # 模型预测
        tracker.start_task("model_prediction")
        tracker.record_memory_usage()
        
        # 创建并训练模型（这里使用基线模型）
        model = HullModel(model_type=args.model_type)
        
        # 注意：在实际应用中，这里应该加载训练好的模型
        # 目前使用简单的随机预测作为演示
        np.random.seed(42)
        predictions = np.random.uniform(0, 2, size=len(test_data))
        
        tracker.record_memory_usage()
        tracker.end_task()
        
        # 创建提交文件
        tracker.start_task("create_submission")
        submission_df = create_submission(predictions, test_data['date_id'])
        
        # 验证提交文件
        if not validate_submission(submission_df):
            return 1
        
        # 保存提交文件 - 根据环境选择输出目录
        if env == "kaggle":
            working_dir_path = Path("/kaggle/working")
        else:
            working_dir_path = log_paths.log_jsonl.parent
        working_dir_path.mkdir(parents=True, exist_ok=True)
        
        kaggle_submission_path = working_dir_path / "submission.csv"
        submission_df.to_csv(kaggle_submission_path, index=False)
        
        # 同时保存parquet格式（如果可能）
        try:
            kaggle_parquet_path = working_dir_path / "submission.parquet"
            submission_df.to_parquet(kaggle_parquet_path, index=False)
        except Exception as e:
            print(f"⚠️  无法保存parquet格式，使用CSV: {e}")
            
        # 也保存到原始路径
        try:
            submission_df.to_csv(submission_path.with_suffix('.csv'), index=False)
            # 同时保存parquet格式（如果可能）
            try:
                submission_df.to_parquet(submission_path, index=False)
            except Exception as e:
                print(f"⚠️  无法保存parquet格式，使用CSV: {e}")
        except Exception as e:
            print(f"⚠️  无法保存到原始路径: {e}")
            
        tracker.end_task()
        
        # 记录性能指标
        tracker.start_task("logging")
        metrics = {
            'num_predictions': len(predictions),
            'min_prediction': float(predictions.min()),
            'max_prediction': float(predictions.max()),
            'mean_prediction': float(predictions.mean()),
            'std_prediction': float(predictions.std()),
            'model_type': args.model_type,
            'environment': env
        }
        
        # 保存日志和指标到Kaggle工作目录
        try:
            kaggle_log_path = Path("/kaggle/working/hull_logs.jsonl")
            kaggle_metrics_path = Path("/kaggle/working/hull_metrics.csv")
            save_logs(tracker.get_summary(), kaggle_log_path)
            save_metrics(metrics, kaggle_metrics_path)
        except Exception as e:
            print(f"⚠️  无法保存日志到Kaggle工作目录: {e}")
            
        # 也保存到原始路径
        try:
            save_logs(tracker.get_summary(), log_paths.log_jsonl)
            save_metrics(metrics, log_paths.metrics_csv)
        except Exception as e:
            print(f"⚠️  无法保存日志到原始路径: {e}")
            
        tracker.end_task()
        
        # 输出结果
        print(f"\n✅ 提交文件已保存: {kaggle_submission_path}")
        print(f"📈 预测统计:")
        print(f"   预测数量: {len(predictions)}")
        print(f"   最小值: {predictions.min():.4f}")
        print(f"   最大值: {predictions.max():.4f}")
        print(f"   平均值: {predictions.mean():.4f}")
        print(f"   标准差: {predictions.std():.4f}")
        
        # 输出性能摘要
        summary = tracker.get_summary()
        print(f"\n⏱️ 性能摘要:")
        print(f"   总时间: {summary['total_time_seconds']:.2f}秒")
        for task, duration in summary['task_breakdown'].items():
            print(f"   {task}: {duration:.2f}秒")
        
        write_result_json(True, result_path)
        return 0
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        write_result_json(False, result_path, error_details=str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
