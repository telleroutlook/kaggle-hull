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
sys.path.insert(0, os.path.dirname(__file__))

from lib.env import detect_run_environment, get_data_paths, get_log_paths
from lib.data import load_test_data, validate_data
from lib.features import engineer_features, get_feature_columns
from lib.models import HullModel, create_submission
from lib.utils import PerformanceTracker, save_logs, save_metrics, validate_submission


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
        "--verbose",
        action="store_true",
        help="详细输出模式"
    )
    
    return parser.parse_args(argv)


def main():
    """主函数 - 运行模型预测"""
    
    args = parse_args()
    tracker = PerformanceTracker()
    
    print("🚀 Hull Tactical - Market Prediction 模型启动")
    print(f"📋 模型类型: {args.model_type}")
    
    # 检测运行环境
    env = detect_run_environment()
    data_paths = get_data_paths(env)
    log_paths = get_log_paths(env)
    
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
        test_data = load_test_data(data_paths)
        
        if not validate_data(test_data, "test"):
            return 1
        
        tracker.end_task()
        
        # 特征工程
        tracker.start_task("feature_engineering")
        feature_cols = get_feature_columns(test_data)
        features = engineer_features(test_data, feature_cols)
        
        if args.verbose:
            print(f"🔧 特征数量: {len(feature_cols)}")
            print(f"📊 特征形状: {features.shape}")
        
        tracker.end_task()
        
        # 模型预测
        tracker.start_task("model_prediction")
        
        # 创建并训练模型（这里使用基线模型）
        model = HullModel(model_type=args.model_type)
        
        # 注意：在实际应用中，这里应该加载训练好的模型
        # 目前使用简单的随机预测作为演示
        np.random.seed(42)
        predictions = np.random.uniform(0, 2, size=len(test_data))
        
        tracker.end_task()
        
        # 创建提交文件
        tracker.start_task("create_submission")
        submission_df = create_submission(predictions, test_data['date_id'])
        
        # 验证提交文件
        if not validate_submission(submission_df):
            return 1
        
        # 保存提交文件
        submission_df.to_csv(submission_path.with_suffix('.csv'), index=False)
        # 同时保存parquet格式（如果可能）
        try:
            submission_df.to_parquet(submission_path, index=False)
        except Exception as e:
            print(f"⚠️  无法保存parquet格式，使用CSV: {e}")
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
        
        # 保存日志和指标
        save_logs(tracker.get_summary(), log_paths.log_jsonl)
        save_metrics(metrics, log_paths.metrics_csv)
        tracker.end_task()
        
        # 输出结果
        print(f"\n✅ 提交文件已保存: {submission_path}")
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
        
        return 0
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())