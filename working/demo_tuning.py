#!/usr/bin/env python3
"""
Hull Tactical - 超参数调优演示和测试脚本
完整的调优流程演示，包括数据分析、模型调优、结果分析和报告生成
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import json
import warnings

warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def run_quick_tuning_demo():
    """运行快速调优演示"""
    print("🚀 Hull Tactical - 超参数调优系统演示")
    print("=" * 60)
    
    # 1. 导入所需模块
    try:
        from hyperparameter_tuning import run_tuning_experiments, TuningConfig
        from tuning_results import TuningResultAnalyzer
        from lib.config import get_config
        from lib.data import load_train_data
        from lib.features import FeaturePipeline
        print("✅ 所有模块导入成功")
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        print("请确保已安装所需依赖: optuna, matplotlib, seaborn, plotly")
        return False
    
    # 2. 检查数据
    data_path = "input/hull-tactical-market-prediction"
    train_file = Path(data_path) / "train.csv"
    
    if not train_file.exists():
        print(f"⚠️ 训练数据文件不存在: {train_file}")
        print("请确保数据文件已正确放置")
        return False
    
    print(f"📁 找到训练数据: {train_file}")
    
    # 3. 创建快速调优配置
    print("\n⚙️ 配置调优参数...")
    config = TuningConfig(
        model_types=["lightgbm", "xgboost"],  # 减少模型数量以加快演示
        n_trials=20,  # 减少试验次数
        cv_folds=3,
        validation_strategy="time_series",
        search_strategy="optuna",
        timeout_seconds=600,  # 10分钟超时
        primary_metric="mse",
        secondary_metrics=["mae"],
        output_dir="demo_tuning_results"
    )
    
    print(f"📊 调优配置:")
    print(f"   模型: {config.model_types}")
    print(f"   试验次数: {config.n_trials}")
    print(f"   验证折数: {config.cv_folds}")
    print(f"   主要指标: {config.primary_metric}")
    
    # 4. 加载和预处理数据
    print("\n📥 加载和预处理数据...")
    try:
        train_data = pd.read_csv(train_file)
        print(f"   原始数据: {train_data.shape}")
        
        # 基础特征工程
        pipeline = FeaturePipeline(stateful=True)
        X = pipeline.fit_transform(train_data)
        y = train_data["forward_returns"].fillna(train_data["forward_returns"].median())
        
        print(f"   特征工程后: {X.shape}")
        print(f"   目标变量统计: 均值={y.mean():.4f}, 标准差={y.std():.4f}")
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return False
    
    # 5. 运行调优
    print("\n🔧 开始超参数调优...")
    start_time = time.time()
    
    try:
        from hyperparameter_tuning import HyperparameterTuner
        
        tuner = HyperparameterTuner(config)
        results = tuner.tune_all_models(X, y)
        
        tuning_time = time.time() - start_time
        print(f"✅ 调优完成! 耗时: {tuning_time:.1f}秒")
        
    except Exception as e:
        print(f"❌ 调优失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. 分析结果
    print("\n📊 分析调优结果...")
    try:
        analyzer = TuningResultAnalyzer("demo_tuning_results")
        analyzer.load_results()
        analyzer.analyze_model_performance()
        summary = analyzer.generate_performance_summary()
        
        print(f"   实验数量: {summary['total_experiments']}")
        print(f"   模型类型: {summary['unique_models']}")
        
        if summary['model_rankings']:
            best_model = min(summary['model_rankings'].keys(), 
                           key=lambda x: summary['model_rankings'][x])
            best_score = summary['model_rankings'][best_model]
            print(f"   🏆 最佳模型: {best_model} (MSE: {best_score:.4f})")
        
    except Exception as e:
        print(f"⚠️ 结果分析失败: {e}")
    
    # 7. 保存调优结果到配置
    print("\n💾 保存调优结果到配置...")
    try:
        config_manager = get_config()
        
        for model_type, result in results.items():
            config_manager.save_tuned_parameters(
                model_type=model_type,
                params=result.best_params,
                best_score=result.best_score
            )
            print(f"   ✅ {model_type} 参数已保存")
        
        print("   💡 建议重新运行主模型以使用优化参数")
        
    except Exception as e:
        print(f"⚠️ 配置保存失败: {e}")
    
    # 8. 生成报告
    print("\n📝 生成分析报告...")
    try:
        chart1 = analyzer.create_performance_comparison_chart()
        chart2 = analyzer.create_model_ranking_chart()
        html_report = analyzer.generate_html_report()
        
        print(f"   📊 性能图表: {chart1}")
        print(f"   🏆 排名图表: {chart2}")
        print(f"   📄 HTML报告: {html_report}")
        
    except Exception as e:
        print(f"⚠️ 报告生成失败: {e}")
    
    # 9. 演示结论
    print("\n" + "=" * 60)
    print("🎉 调优演示完成!")
    print("=" * 60)
    
    print(f"\n📋 总结:")
    print(f"   ⏱️  总耗时: {tuning_time:.1f}秒")
    print(f"   🧪 试验总数: {sum(r.n_trials for r in results.values())}")
    print(f"   🏆 最佳模型: {best_model if 'best_model' in locals() else 'N/A'}")
    
    if results:
        print(f"\n📈 各模型性能:")
        for model_type, result in results.items():
            improvement = ""
            print(f"   {model_type}: MSE={result.best_score:.4f}, 试验数={result.n_trials}")
    
    print(f"\n📁 输出文件:")
    print(f"   调优结果: demo_tuning_results/")
    print(f"   HTML报告: {html_report if 'html_report' in locals() else '生成失败'}")
    
    print(f"\n💡 下一步建议:")
    print(f"   1. 查看HTML报告了解详细分析")
    print(f"   2. 运行 python main.py 测试优化后的模型")
    print(f"   3. 根据需要调整调优参数进行深度优化")
    
    return True


def test_optimized_models():
    """测试优化后的模型性能"""
    print("\n🧪 测试优化后的模型性能...")
    
    try:
        from lib.model_registry import get_model_params
        from lib.config import get_config
        
        config = get_config()
        
        print("📊 当前模型参数配置:")
        for model_type in ["lightgbm", "xgboost", "catboost"]:
            params = get_model_params(model_type)
            is_tuned = config.is_tuning_enabled() and config.get_tuned_parameters(model_type)
            status = "🎯 已调优" if is_tuned else "📋 默认"
            print(f"   {model_type}: {status} ({len(params)}个参数)")
        
        print("\n✅ 配置系统工作正常!")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def create_usage_examples():
    """创建使用示例"""
    print("\n📚 创建使用示例...")
    
    examples_dir = Path("tuning_examples")
    examples_dir.mkdir(exist_ok=True)
    
    # 1. 基础调优示例
    basic_example = '''#!/usr/bin/env python3
"""
基础超参数调优示例
"""

from hyperparameter_tuning import run_tuning_experiments

# 运行基础调优
tuner, results = run_tuning_experiments(
    data_path="input/hull-tactical-market-prediction",
    output_dir="my_tuning_results"
)

print(f"最佳模型: {tuner.get_ranking()[0][0]}")
'''
    
    with open(examples_dir / "basic_tuning.py", 'w') as f:
        f.write(basic_example)
    
    # 2. 自定义配置示例
    custom_example = '''#!/usr/bin/env python3
"""
自定义配置调优示例
"""

from hyperparameter_tuning import TuningConfig, HyperparameterTuner
from lib.data import load_train_data
from lib.features import FeaturePipeline

# 自定义配置
config = TuningConfig(
    model_types=["lightgbm", "xgboost", "catboost"],
    n_trials=100,
    cv_folds=5,
    search_strategy="optuna",
    validation_strategy="time_series",
    primary_metric="mse",
    secondary_metrics=["mae", "r2"],
    timeout_seconds=3600
)

# 加载数据
train_data = load_train_data("input/hull-tactical-market-prediction")
pipeline = FeaturePipeline(stateful=True)
X = pipeline.fit_transform(train_data)
y = train_data["forward_returns"].fillna(train_data["forward_returns"].median())

# 运行调优
tuner = HyperparameterTuner(config)
results = tuner.tune_all_models(X, y)

# 保存结果
tuner.save_results("my_custom_tuning.json")
'''
    
    with open(examples_dir / "custom_tuning.py", 'w') as f:
        f.write(custom_example)
    
    # 3. 结果分析示例
    analysis_example = '''#!/usr/bin/env python3
"""
调优结果分析示例
"""

from tuning_results import TuningResultAnalyzer

# 创建分析器
analyzer = TuningResultAnalyzer("tuning_results")

# 加载和分析结果
analyzer.load_results()
analyzer.analyze_model_performance()
summary = analyzer.generate_performance_summary()

# 生成报告
chart1 = analyzer.create_performance_comparison_chart()
chart2 = analyzer.create_model_ranking_chart()
html_report = analyzer.generate_html_report()

print(f"报告生成完成: {html_report}")
'''
    
    with open(examples_dir / "analyze_results.py", 'w') as f:
        f.write(analysis_example)
    
    print(f"✅ 示例文件已创建在: {examples_dir}/")
    print("   basic_tuning.py - 基础调优")
    print("   custom_tuning.py - 自定义配置")
    print("   analyze_results.py - 结果分析")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hull Tactical 调优演示")
    parser.add_argument("--quick", action="store_true", help="快速演示")
    parser.add_argument("--test", action="store_true", help="测试配置")
    parser.add_argument("--examples", action="store_true", help="创建示例")
    
    args = parser.parse_args()
    
    if not any([args.quick, args.test, args.examples]):
        args.quick = True  # 默认运行快速演示
    
    success = True
    
    if args.test:
        success &= test_optimized_models()
    
    if args.examples:
        create_usage_examples()
    
    if args.quick:
        success &= run_quick_tuning_demo()
    
    if success:
        print("\n🎉 所有操作完成!")
    else:
        print("\n❌ 某些操作失败，请检查错误信息")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())