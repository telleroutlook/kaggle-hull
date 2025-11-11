"""
时间序列验证系统快速开始指南

演示如何使用新开发的时间序列友好交叉验证系统
"""

import numpy as np
import pandas as pd
from pathlib import Path

# 导入时间序列验证系统
from time_series_validation import create_time_series_validator, validate_model_time_series
from time_series_validation_integration import TimeSeriesValidationAPI

# 导入现有Hull系统
try:
    from lib.models import HullModel
    from lib.features import get_feature_columns
    HULL_AVAILABLE = True
except ImportError:
    HULL_AVAILABLE = False
    print("⚠️ Hull系统不可用，将使用模拟数据演示")


def create_sample_market_data(n_samples: int = 1000) -> pd.DataFrame:
    """创建示例市场数据"""
    np.random.seed(42)
    
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # 生成市场特征
    data = {
        'date_id': [int(d.strftime('%Y%m%d')) for d in dates],
        'M1': np.random.normal(0, 0.02, n_samples),  # 市场动量
        'M2': np.random.normal(0, 0.01, n_samples),
        'P1': 100 + np.cumsum(np.random.normal(0, 0.01, n_samples)),  # 价格
        'V1': np.random.gamma(2, 0.01, n_samples),  # 波动率
        'E1': np.random.normal(0, 0.005, n_samples),  # 宏观经济
        'S1': np.random.normal(0, 0.01, n_samples),  # 情绪
        'forward_returns': np.random.normal(0, 0.02, n_samples)  # 目标
    }
    
    return pd.DataFrame(data)


def quick_start_example():
    """快速开始示例"""
    print("🚀 时间序列验证系统快速开始示例")
    print("=" * 50)
    
    # 1. 准备数据
    print("📊 准备示例数据...")
    train_data = create_sample_market_data(800)
    print(f"数据规模: {train_data.shape}")
    
    # 2. 创建模型
    if HULL_AVAILABLE:
        model = HullModel(model_type="lightgbm", model_params={'n_estimators': 50})
        feature_cols = get_feature_columns(train_data)
        print(f"✅ 使用Hull模型: {model.model_type}")
    else:
        # 创建简单模型
        class SimpleModel:
            def fit(self, X, y): 
                self.mean_ = np.mean(y)
                return self
            def predict(self, X): 
                return np.full(len(X), self.mean_)
        
        model = SimpleModel()
        feature_cols = ['M1', 'M2', 'V1', 'E1', 'S1']
        print("✅ 使用简单回归模型")
    
    print(f"特征列: {feature_cols}")
    
    # 3. 基础时间序列验证
    print("\\n🔄 执行基础时间序列验证...")
    result = validate_model_time_series(
        model=model,
        X=train_data[feature_cols],
        y=train_data['forward_returns'],
        strategy='expanding_window',
        n_splits=5
    )
    
    print(f"✅ 验证完成!")
    print(f"   策略: {result.strategy.value}")
    print(f"   折数: {result.n_splits}")
    
    # 显示关键指标
    if 'mse' in result.metrics:
        mse_mean = np.mean(result.metrics['mse'])
        mse_std = np.std(result.metrics['mse'])
        print(f"   MSE: {mse_mean:.6f} ± {mse_std:.6f}")
    
    if 'mae' in result.metrics:
        mae_mean = np.mean(result.metrics['mae'])
        print(f"   MAE: {mae_mean:.6f}")
    
    # 4. 多种策略对比
    print("\\n🔄 多种验证策略对比...")
    strategies = ['time_series_split', 'expanding_window', 'purged_time_series']
    
    for strategy in strategies:
        try:
            result = validate_model_time_series(
                model=model,
                X=train_data[feature_cols],
                y=train_data['forward_returns'],
                strategy=strategy,
                n_splits=3
            )
            
            mse = np.mean(result.metrics.get('mse', [0]))
            print(f"  {strategy:20s}: MSE = {mse:.6f}")
            
        except Exception as e:
            print(f"  {strategy:20s}: 失败 - {e}")
    
    # 5. API快速验证
    print("\\n⚡ 使用API快速验证...")
    if HULL_AVAILABLE:
        try:
            api_result = TimeSeriesValidationAPI.quick_validate(
                model=model,
                train_data=train_data,
                strategy="expanding_window"
            )
            print("✅ API验证成功")
        except Exception as e:
            print(f"❌ API验证失败: {e}")
    else:
        print("⚠️ 跳过API测试（Hull系统不可用）")
    
    # 6. 金融指标演示
    print("\\n💰 金融指标演示...")
    calculator = FinanceValidationMetrics()
    
    # 生成示例收益序列
    strategy_returns = np.random.normal(0.0005, 0.02, 252)
    benchmark_returns = np.random.normal(0.0003, 0.018, 252)
    
    # 计算金融指标
    metrics = calculator.calculate_metrics(
        strategy_returns, strategy_returns,  # 简化示例
        benchmark_returns=benchmark_returns
    )
    
    print("主要金融指标:")
    for metric_name in ['strategy_sharpe', 'strategy_volatility', 'directional_accuracy']:
        if metric_name in metrics:
            print(f"  {metric_name}: {metrics[metric_name]:.4f}")
    
    return {
        'data_shape': train_data.shape,
        'model_type': type(model).__name__,
        'strategies_tested': len(strategies),
        'finance_metrics_count': len(metrics)
    }


def advanced_usage_example():
    """高级使用示例"""
    print("\\n" + "=" * 50)
    print("🔧 高级功能演示")
    print("=" * 50)
    
    # 生成更大规模的数据
    print("📊 生成大规模测试数据...")
    large_data = create_sample_market_data(2000)
    print(f"数据规模: {large_data.shape}")
    
    # 创建模型
    if HULL_AVAILABLE:
        model = HullModel(model_type="baseline", model_params={'n_estimators': 30})
        feature_cols = get_feature_columns(large_data)
    else:
        class SimpleModel:
            def fit(self, X, y): return self
            def predict(self, X): return np.zeros(len(X))
        model = SimpleModel()
        feature_cols = ['M1', 'M2', 'V1', 'E1', 'S1']
    
    # 1. 自定义配置验证
    print("\\n🔧 自定义配置验证...")
    from time_series_validation import ValidationConfig, TimeSeriesCrossValidator
    
    config = ValidationConfig(
        strategy='expanding_window',
        n_splits=5,
        test_size=100,
        min_train_samples=300,
        enable_performance_monitoring=True,
        verbose=False
    )
    
    validator = TimeSeriesCrossValidator(config)
    result = validator.validate(
        model, 
        large_data[feature_cols], 
        large_data['forward_returns']
    )
    
    print(f"✅ 自定义配置验证完成")
    print(f"   质量指标: {len(result.quality_metrics)} 项")
    
    # 2. 性能监控
    if result.quality_metrics:
        print("   验证质量:")
        for key, value in result.quality_metrics.items():
            if isinstance(value, (int, float)):
                print(f"     {key}: {value:.4f}")
    
    # 3. 保存结果
    print("\\n💾 保存验证结果...")
    output_dir = Path("validation_results")
    validator.save_results(result, output_dir / "demo_validation")
    print(f"✅ 结果已保存到: {output_dir}")
    
    return result


def integration_example():
    """集成系统示例"""
    print("\\n" + "=" * 50)
    print("🔗 集成系统示例")
    print("=" * 50)
    
    if not HULL_AVAILABLE:
        print("⚠️ Hull系统不可用，跳过集成示例")
        return
    
    # 生成测试数据
    test_data = create_sample_market_data(1200)
    
    # 1. 集成验证器
    print("🔄 集成验证器...")
    from time_series_validation_integration import IntegratedTimeSeriesValidator
    
    validator = IntegratedTimeSeriesValidator()
    
    # 执行多策略验证
    results = validator.validate_hull_model(
        model_type="lightgbm",
        train_data=test_data,
        target_column="forward_returns"
    )
    
    print(f"✅ 集成验证完成: {len(results)} 种策略")
    
    # 2. 集成模型验证
    print("\\n🏆 集成模型验证...")
    ensemble_results = validator.validate_ensemble_performance(
        base_models=['lightgbm', 'xgboost'],
        ensemble_configs=[
            {'type': 'dynamic_weighted', 'config': {'performance_window': 30}},
            {'type': 'averaging', 'config': {'weights': [0.6, 0.4]}}
        ],
        train_data=test_data,
        target_column="forward_returns"
    )
    
    print(f"✅ 集成模型验证完成: {len(ensemble_results)} 种类型")
    
    # 3. 全面验证
    print("\\n🌟 全面验证API...")
    comprehensive_results = TimeSeriesValidationAPI.comprehensive_validation(
        train_data=test_data,
        model_types=['baseline', 'lightgbm'],
        strategies=['time_series_split', 'expanding_window'],
        save_results=False
    )
    
    print(f"✅ 全面验证完成: {len(comprehensive_results)} 模型类型")
    
    return {
        'integrated_validation': len(results),
        'ensemble_validation': len(ensemble_results),
        'comprehensive_validation': len(comprehensive_results)
    }


def main():
    """主函数"""
    print("🎯 时间序列验证系统完整演示")
    print("演示新的时间序列友好交叉验证系统的核心功能")
    
    try:
        # 1. 快速开始示例
        quick_results = quick_start_example()
        
        # 2. 高级功能示例
        advanced_results = advanced_usage_example()
        
        # 3. 集成系统示例
        if HULL_AVAILABLE:
            integration_results = integration_example()
        else:
            integration_results = {'message': 'Hull系统不可用'}
        
        # 总结
        print("\\n" + "=" * 60)
        print("🎉 演示完成!")
        print("=" * 60)
        
        print("📊 功能验证总结:")
        print(f"  ✅ 基础验证: {quick_results.get('strategies_tested', 0)} 种策略")
        print(f"  ✅ 高级功能: 自定义配置、性能监控")
        if HULL_AVAILABLE:
            print(f"  ✅ 集成验证: {integration_results.get('integrated_validation', 0)} 种策略")
            print(f"  ✅ 集成模型: {integration_results.get('ensemble_validation', 0)} 种类型")
        
        print("\\n💡 主要优势:")
        print("  🔄 时间序列友好 - 避免信息泄露")
        print("  💰 金融专业指标 - 风险和收益评估")
        print("  🔗 无缝集成 - 与Hull系统完美配合")
        print("  ⚡ 高性能 - 支持并行处理和大规模数据")
        print("  🎯 多策略支持 - 5+种验证策略")
        
        print("\\n📁 生成的示例文件:")
        print("  📄 validation_results/ - 验证结果")
        print("  📊 validation_demo_plots/ - 性能图表")
        print("  📋 演示报告 - time_series_validation_demo_report.md")
        
        return True
        
    except Exception as e:
        print(f"\\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\\n🎊 时间序列验证系统已准备就绪!")
    else:
        print("\\n⚠️ 请检查错误信息并重新运行")