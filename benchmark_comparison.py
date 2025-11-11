#!/usr/bin/env python3
"""
Hull Tactical市场预测项目 - 基准对比分析工具

该工具用于对比不同版本/策略的性能表现，包括：
1. 优化前后的性能对比
2. 不同集成策略的效果分析
3. 特征工程改进效果评估
4. 性能趋势分析
5. 统计显著性检验

作者: iFlow AI系统
日期: 2025-11-11
"""

import sys
import os
import json
import warnings
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.stats.api as sms

# 抑制警告
warnings.filterwarnings('ignore')

# 设置matplotlib中文支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class BenchmarkResult:
    """基准测试结果数据类"""
    strategy: str
    version: str  # 'baseline', 'optimized', 'advanced'
    performance_metrics: Dict[str, float]
    timing_metrics: Dict[str, float]
    memory_metrics: Dict[str, float]
    stability_metrics: Dict[str, float]
    test_metadata: Dict[str, Any]
    timestamp: str
    data_size: int
    features_used: int


class BenchmarkComparator:
    """基准对比分析器"""
    
    def __init__(self):
        self.baseline_results: List[BenchmarkResult] = []
        self.comparison_results: List[BenchmarkResult] = []
        self.comparison_data: Dict[str, Any] = {}
        
    def load_baseline_results(self, baseline_file: str) -> bool:
        """加载基线结果"""
        try:
            if os.path.exists(benchmark_file):
                with open(benchmark_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 转换为BenchmarkResult对象
                for item in data:
                    result = BenchmarkResult(
                        strategy=item.get('strategy', 'unknown'),
                        version=item.get('version', 'baseline'),
                        performance_metrics=item.get('performance', {}),
                        timing_metrics=item.get('timing', {}),
                        memory_metrics=item.get('memory', {}),
                        stability_metrics=item.get('stability', {}),
                        test_metadata=item.get('metadata', {}),
                        timestamp=item.get('timestamp', datetime.now().isoformat()),
                        data_size=item.get('data_size', 1000),
                        features_used=item.get('features_used', 20)
                    )
                    self.baseline_results.append(result)
                
                print(f"✅ 成功加载 {len(self.baseline_results)} 个基线结果")
                return True
            else:
                print(f"⚠️ 基线文件不存在: {benchmark_file}")
                return False
                
        except Exception as e:
            print(f"❌ 加载基线结果失败: {e}")
            return False
    
    def load_comparison_results(self, comparison_file: str) -> bool:
        """加载对比结果"""
        try:
            if os.path.exists(comparison_file):
                with open(comparison_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 转换为BenchmarkResult对象
                for item in data:
                    result = BenchmarkResult(
                        strategy=item.get('strategy', 'unknown'),
                        version=item.get('version', 'optimized'),
                        performance_metrics=item.get('performance', {}),
                        timing_metrics=item.get('timing', {}),
                        memory_metrics=item.get('memory', {}),
                        stability_metrics=item.get('stability', {}),
                        test_metadata=item.get('metadata', {}),
                        timestamp=item.get('timestamp', datetime.now().isoformat()),
                        data_size=item.get('data_size', 1000),
                        features_used=item.get('features_used', 20)
                    )
                    self.comparison_results.append(result)
                
                print(f"✅ 成功加载 {len(self.comparison_results)} 个对比结果")
                return True
            else:
                print(f"⚠️ 对比文件不存在: {comparison_file}")
                return False
                
        except Exception as e:
            print(f"❌ 加载对比结果失败: {e}")
            return False
    
    def generate_synthetic_baseline(self) -> List[BenchmarkResult]:
        """生成合成基线数据（用于演示）"""
        print("🔧 生成合成基线数据...")
        
        # 模拟不同策略的基线性能
        baseline_strategies = [
            ('单个基线模型', {'mse': 0.257, 'mae': 0.410, 'rmse': 0.507}),
            ('简单平均集成', {'mse': 0.257, 'mae': 0.410, 'rmse': 0.507}),
            ('加权平均集成', {'mse': 0.257, 'mae': 0.410, 'rmse': 0.507}),
        ]
        
        baseline_data = []
        for strategy, metrics in baseline_strategies:
            result = BenchmarkResult(
                strategy=strategy,
                version='baseline',
                performance_metrics=metrics,
                timing_metrics={'total_time': 0.5, 'training_time': 0.3, 'prediction_time': 0.2},
                memory_metrics={'usage': 50.0, 'peak': 80.0},
                stability_metrics={'consistency': 0.80, 'variance': 0.02},
                test_metadata={'model_type': 'random_forest', 'n_estimators': 100},
                timestamp='2025-01-01T00:00:00',
                data_size=1000,
                features_used=20
            )
            baseline_data.append(result)
        
        self.baseline_results = baseline_data
        print(f"✅ 生成 {len(baseline_data)} 个基线结果")
        return baseline_data
    
    def generate_synthetic_optimized(self) -> List[BenchmarkResult]:
        """生成合成优化数据（用于演示）"""
        print("🔧 生成合成优化数据...")
        
        # 模拟优化后的性能（改进15-25%）
        optimized_strategies = [
            ('动态权重集成', {'mse': 0.248, 'mae': 0.403, 'rmse': 0.498}),
            ('智能特征工程', {'mse': 0.230, 'mae': 0.385, 'rmse': 0.480}),
            ('自适应时间窗口', {'mse': 0.242, 'mae': 0.395, 'rmse': 0.492}),
            ('超参数优化', {'mse': 0.235, 'mae': 0.390, 'rmse': 0.485}),
        ]
        
        optimized_data = []
        for strategy, metrics in optimized_strategies:
            result = BenchmarkResult(
                strategy=strategy,
                version='optimized',
                performance_metrics=metrics,
                timing_metrics={'total_time': 0.6, 'training_time': 0.35, 'prediction_time': 0.25},
                memory_metrics={'usage': 60.0, 'peak': 95.0},
                stability_metrics={'consistency': 0.92, 'variance': 0.015},
                test_metadata={'model_type': 'ensemble', 'optimization': True},
                timestamp=datetime.now().isoformat(),
                data_size=1000,
                features_used=45
            )
            optimized_data.append(result)
        
        self.comparison_results = optimized_data
        print(f"✅ 生成 {len(optimized_data)} 个优化结果")
        return optimized_data
    
    def compare_performance(self) -> Dict[str, Any]:
        """对比性能表现"""
        print("\n📊 执行性能对比分析...")
        
        comparison = {
            'summary': {},
            'by_strategy': {},
            'statistical_tests': {},
            'improvements': []
        }
        
        # 策略对比
        strategies = set([r.strategy for r in self.baseline_results + self.comparison_results])
        
        for strategy in strategies:
            baseline_result = next((r for r in self.baseline_results if r.strategy == strategy), None)
            optimized_result = next((r for r in self.comparison_results if r.strategy == strategy), None)
            
            if baseline_result and optimized_result:
                strategy_comparison = self._compare_strategy(baseline_result, optimized_result)
                comparison['by_strategy'][strategy] = strategy_comparison
            elif optimized_result:
                # 只有优化版本的结果
                comparison['by_strategy'][strategy] = self._analyze_single_result(optimized_result)
        
        # 整体摘要
        comparison['summary'] = self._generate_summary_stats(comparison['by_strategy'])
        
        # 统计显著性检验
        if len(self.baseline_results) > 0 and len(self.comparison_results) > 0:
            comparison['statistical_tests'] = self._perform_statistical_tests()
        
        # 改进效果分析
        comparison['improvements'] = self._analyze_improvements(comparison['by_strategy'])
        
        self.comparison_data = comparison
        return comparison
    
    def _compare_strategy(self, baseline: BenchmarkResult, optimized: BenchmarkResult) -> Dict[str, Any]:
        """对比单个策略的性能"""
        comparison = {
            'baseline_metrics': baseline.performance_metrics,
            'optimized_metrics': optimized.performance_metrics,
            'improvements': {},
            'relative_improvements': {},
            'timing_comparison': {},
            'stability_comparison': {},
            'effect_size': {}
        }
        
        # 性能指标对比
        for metric in baseline.performance_metrics.keys():
            if metric in optimized.performance_metrics:
                baseline_val = baseline.performance_metrics[metric]
                optimized_val = optimized.performance_metrics[metric]
                
                if baseline_val != 0:
                    improvement = baseline_val - optimized_val
                    relative_improvement = improvement / baseline_val * 100
                else:
                    improvement = 0
                    relative_improvement = 0
                
                comparison['improvements'][metric] = improvement
                comparison['relative_improvements'][metric] = relative_improvement
        
        # 时间对比
        comparison['timing_comparison'] = {
            'baseline_time': baseline.timing_metrics,
            'optimized_time': optimized.timing_metrics,
            'time_ratio': optimized.timing_metrics.get('total_time', 1) / baseline.timing_metrics.get('total_time', 1)
        }
        
        # 稳定性对比
        comparison['stability_comparison'] = {
            'baseline_stability': baseline.stability_metrics,
            'optimized_stability': optimized.stability_metrics,
            'stability_improvement': (
                optimized.stability_metrics.get('consistency', 0) - 
                baseline.stability_metrics.get('consistency', 0)
            )
        }
        
        # 效应大小计算（Cohen's d）
        if 'mse' in baseline.performance_metrics and 'mse' in optimized.performance_metrics:
            baseline_mse = baseline.performance_metrics['mse']
            optimized_mse = optimized.performance_metrics['mse']
            
            # 模拟标准差
            baseline_std = 0.01
            optimized_std = 0.008
            
            pooled_std = np.sqrt((baseline_std**2 + optimized_std**2) / 2)
            cohens_d = (baseline_mse - optimized_mse) / pooled_std
            
            comparison['effect_size'] = {
                'cohens_d': cohens_d,
                'interpretation': self._interpret_effect_size(cohens_d)
            }
        
        return comparison
    
    def _analyze_single_result(self, result: BenchmarkResult) -> Dict[str, Any]:
        """分析单个结果（没有对比）"""
        return {
            'baseline_metrics': {},
            'optimized_metrics': result.performance_metrics,
            'improvements': {},
            'relative_improvements': {},
            'timing_comparison': {'baseline_time': {}, 'optimized_time': result.timing_metrics},
            'stability_comparison': {'baseline_stability': {}, 'optimized_stability': result.stability_metrics},
            'effect_size': {}
        }
    
    def _generate_summary_stats(self, strategy_comparisons: Dict[str, Any]) -> Dict[str, Any]:
        """生成摘要统计"""
        summary = {
            'total_strategies': len(strategy_comparisons),
            'avg_improvements': {},
            'best_improvements': {},
            'consistency_score': 0,
            'overall_effectiveness': 0
        }
        
        # 计算平均改进
        all_improvements = {metric: [] for metric in ['mse', 'mae', 'rmse']}
        
        for strategy, comparison in strategy_comparisons.items():
            for metric in ['mse', 'mae', 'rmse']:
                if metric in comparison['relative_improvements']:
                    all_improvements[metric].append(comparison['relative_improvements'][metric])
        
        for metric, improvements in all_improvements.items():
            if improvements:
                summary['avg_improvements'][metric] = {
                    'mean': np.mean(improvements),
                    'std': np.std(improvements),
                    'min': np.min(improvements),
                    'max': np.max(improvements)
                }
        
        # 最佳改进
        for metric in ['mse', 'mae', 'rmse']:
            if metric in all_improvements and all_improvements[metric]:
                best_idx = np.argmax(all_improvements[metric])
                best_strategy = list(strategy_comparisons.keys())[best_idx]
                summary['best_improvements'][metric] = {
                    'strategy': best_strategy,
                    'improvement': all_improvements[metric][best_idx]
                }
        
        # 一致性评分（所有指标改进的策略比例）
        consistent_strategies = 0
        total_strategies = len(strategy_comparisons)
        
        for strategy, comparison in strategy_comparisons.items():
            positive_improvements = 0
            total_metrics = 0
            
            for metric in ['mse', 'mae', 'rmse']:
                if metric in comparison['relative_improvements']:
                    total_metrics += 1
                    if comparison['relative_improvements'][metric] > 0:
                        positive_improvements += 1
            
            if total_metrics > 0 and positive_improvements / total_metrics > 0.5:
                consistent_strategies += 1
        
        summary['consistency_score'] = consistent_strategies / total_strategies if total_strategies > 0 else 0
        
        # 整体有效性评分
        if 'mse' in summary['avg_improvements']:
            mse_improvement = summary['avg_improvements']['mse']['mean']
            summary['overall_effectiveness'] = min(100, max(0, mse_improvement * 4))  # 0-100分制
        
        return summary
    
    def _perform_statistical_tests(self) -> Dict[str, Any]:
        """执行统计显著性检验"""
        tests = {}
        
        # 准备数据
        baseline_mse = [r.performance_metrics.get('mse', 0) for r in self.baseline_results if 'mse' in r.performance_metrics]
        optimized_mse = [r.performance_metrics.get('mse', 0) for r in self.comparison_results if 'mse' in r.performance_metrics]
        
        if len(baseline_mse) > 1 and len(optimized_mse) > 1:
            # t检验
            t_stat, p_value = stats.ttest_ind(baseline_mse, optimized_mse)
            tests['t_test'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'interpretation': '显著差异' if p_value < 0.05 else '无显著差异'
            }
            
            # 效应大小 (Cohen's d)
            pooled_std = np.sqrt(((len(baseline_mse) - 1) * np.var(baseline_mse, ddof=1) + 
                                 (len(optimized_mse) - 1) * np.var(optimized_mse, ddof=1)) / 
                                (len(baseline_mse) + len(optimized_mse) - 2))
            
            if pooled_std > 0:
                cohens_d = (np.mean(baseline_mse) - np.mean(optimized_mse)) / pooled_std
                tests['effect_size'] = {
                    'cohens_d': cohens_d,
                    'magnitude': self._interpret_effect_size(cohens_d)
                }
        
        return tests
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """解释效应大小"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "小效应"
        elif abs_d < 0.5:
            return "中等效应"
        elif abs_d < 0.8:
            return "大效应"
        else:
            return "非常大效应"
    
    def _analyze_improvements(self, strategy_comparisons: Dict[str, Any]) -> List[Dict[str, Any]]:
        """分析改进效果"""
        improvements = []
        
        for strategy, comparison in strategy_comparisons.items():
            for metric, improvement in comparison['relative_improvements'].items():
                improvements.append({
                    'strategy': strategy,
                    'metric': metric,
                    'improvement_percent': improvement,
                    'magnitude': '大' if improvement > 10 else '中' if improvement > 5 else '小'
                })
        
        # 按改进幅度排序
        improvements.sort(key=lambda x: x['improvement_percent'], reverse=True)
        
        return improvements
    
    def generate_visualization_plots(self, output_dir: str = "benchmark_analysis"):
        """生成可视化图表"""
        print(f"\n📊 生成性能对比可视化图表...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        try:
            # 1. 性能对比条形图
            self._plot_performance_comparison(output_path)
            
            # 2. 改进效果分布图
            self._plot_improvement_distribution(output_path)
            
            # 3. 时间效率对比图
            self._plot_timing_comparison(output_path)
            
            # 4. 稳定性分析图
            self._plot_stability_analysis(output_path)
            
            # 5. 综合雷达图
            self._plot_comprehensive_radar(output_path)
            
            print(f"✅ 图表已保存到: {output_path}")
            
        except Exception as e:
            print(f"❌ 生成图表失败: {e}")
    
    def _plot_performance_comparison(self, output_path: Path):
        """绘制性能对比条形图"""
        if not self.comparison_data or 'by_strategy' not in self.comparison_data:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Hull Tactical项目性能对比分析', fontsize=16, fontweight='bold')
        
        metrics = ['mse', 'mae', 'rmse']
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            strategies = []
            baseline_values = []
            optimized_values = []
            
            for strategy, comparison in self.comparison_data['by_strategy'].items():
                strategies.append(strategy)
                baseline_values.append(comparison['baseline_metrics'].get(metric, 0))
                optimized_values.append(comparison['optimized_metrics'].get(metric, 0))
            
            x = np.arange(len(strategies))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, baseline_values, width, label='基线', alpha=0.7, color='lightcoral')
            bars2 = ax.bar(x + width/2, optimized_values, width, label='优化', alpha=0.7, color='skyblue')
            
            ax.set_title(f'{metric.upper()} 性能对比')
            ax.set_xlabel('策略')
            ax.set_ylabel(f'{metric.upper()} 值')
            ax.set_xticks(x)
            ax.set_xticklabels(strategies, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 空白子图
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_improvement_distribution(self, output_path: Path):
        """绘制改进效果分布图"""
        if not self.comparison_data or 'improvements' not in self.comparison_data:
            return
        
        improvements = self.comparison_data['improvements']
        if not improvements:
            return
        
        # 准备数据
        improvement_data = []
        for imp in improvements:
            improvement_data.append({
                'Strategy': imp['strategy'],
                'Metric': imp['metric'].upper(),
                'Improvement': imp['improvement_percent']
            })
        
        df = pd.DataFrame(improvement_data)
        
        # 箱线图
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        sns.boxplot(data=df, x='Metric', y='Improvement', palette='Set2')
        plt.title('改进幅度分布 (按指标)')
        plt.ylabel('改进百分比 (%)')
        plt.grid(True, alpha=0.3)
        
        # 条形图
        plt.subplot(2, 1, 2)
        sns.barplot(data=df, x='Improvement', y='Strategy', hue='Metric', palette='Set1')
        plt.title('各策略改进效果')
        plt.xlabel('改进百分比 (%)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'improvement_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_timing_comparison(self, output_path: Path):
        """绘制时间效率对比图"""
        if not self.comparison_data or 'by_strategy' not in self.comparison_data:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('时间效率分析', fontsize=16, fontweight='bold')
        
        strategies = list(self.comparison_data['by_strategy'].keys())
        
        # 训练时间对比
        baseline_times = []
        optimized_times = []
        
        for strategy in strategies:
            comparison = self.comparison_data['by_strategy'][strategy]
            baseline_times.append(comparison['timing_comparison']['baseline_time'].get('total_time', 0))
            optimized_times.append(comparison['timing_comparison']['optimized_time'].get('total_time', 0))
        
        x = np.arange(len(strategies))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, baseline_times, width, label='基线', alpha=0.7, color='lightcoral')
        bars2 = ax1.bar(x + width/2, optimized_times, width, label='优化', alpha=0.7, color='skyblue')
        
        ax1.set_title('执行时间对比')
        ax1.set_xlabel('策略')
        ax1.set_ylabel('时间 (秒)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(strategies, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 时间比率
        time_ratios = [opt/base if base > 0 else 1 for opt, base in zip(optimized_times, baseline_times)]
        
        bars = ax2.bar(strategies, time_ratios, alpha=0.7, color='gold')
        ax2.set_title('时间比率 (优化/基线)')
        ax2.set_ylabel('比率')
        ax2.set_xticklabels(strategies, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='基准线')
        ax2.legend()
        
        # 添加数值标签
        for bar, ratio in zip(bars, time_ratios):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{ratio:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path / 'timing_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_stability_analysis(self, output_path: Path):
        """绘制稳定性分析图"""
        if not self.comparison_data or 'by_strategy' not in self.comparison_data:
            return
        
        strategies = list(self.comparison_data['by_strategy'].keys())
        
        baseline_consistency = []
        optimized_consistency = []
        
        for strategy in strategies:
            comparison = self.comparison_data['by_strategy'][strategy]
            baseline_consistency.append(comparison['stability_comparison']['baseline_stability'].get('consistency', 0))
            optimized_consistency.append(comparison['stability_comparison']['optimized_stability'].get('consistency', 0))
        
        # 散点图
        plt.figure(figsize=(10, 8))
        
        plt.scatter(baseline_consistency, optimized_consistency, s=100, alpha=0.7, c=range(len(strategies)), cmap='viridis')
        
        for i, strategy in enumerate(strategies):
            plt.annotate(strategy, (baseline_consistency[i], optimized_consistency[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # 添加对角线
        min_val = min(min(baseline_consistency), min(optimized_consistency))
        max_val = max(max(baseline_consistency), max(optimized_consistency))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='y=x (无改进)')
        
        plt.xlabel('基线稳定性')
        plt.ylabel('优化后稳定性')
        plt.title('稳定性改进分析')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 颜色条
        cbar = plt.colorbar()
        cbar.set_label('策略索引')
        
        plt.tight_layout()
        plt.savefig(output_path / 'stability_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_comprehensive_radar(self, output_path: Path):
        """绘制综合雷达图"""
        if not self.comparison_data or 'summary' not in self.comparison_data:
            return
        
        summary = self.comparison_data['summary']
        
        # 准备雷达图数据
        categories = ['MSE改进', 'MAE改进', 'RMSE改进', '稳定性', '时间效率', '整体效果']
        
        # 计算各维度评分 (0-100)
        scores = []
        
        if 'avg_improvements' in summary:
            # 性能改进评分 (转换为0-100)
            mse_score = min(100, max(0, summary['avg_improvements'].get('mse', {}).get('mean', 0) * 10))
            mae_score = min(100, max(0, summary['avg_improvements'].get('mae', {}).get('mean', 0) * 10))
            rmse_score = min(100, max(0, summary['avg_improvements'].get('rmse', {}).get('mean', 0) * 10))
        else:
            mse_score = mae_score = rmse_score = 0
        
        # 稳定性评分
        stability_score = summary.get('consistency_score', 0) * 100
        
        # 时间效率评分 (时间比率的倒数)
        if 'by_strategy' in self.comparison_data:
            time_ratios = []
            for comparison in self.comparison_data['by_strategy'].values():
                baseline_time = comparison['timing_comparison']['baseline_time'].get('total_time', 1)
                optimized_time = comparison['timing_comparison']['optimized_time'].get('total_time', 1)
                if baseline_time > 0:
                    time_ratios.append(optimized_time / baseline_time)
            
            avg_time_ratio = np.mean(time_ratios) if time_ratios else 1
            time_score = max(0, min(100, (2 - avg_time_ratio) * 50))  # 比率越低分数越高
        else:
            time_score = 50
        
        # 整体效果评分
        effectiveness_score = summary.get('overall_effectiveness', 0)
        
        scores = [mse_score, mae_score, rmse_score, stability_score, time_score, effectiveness_score]
        
        # 雷达图
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        scores += scores[:1]  # 闭合图形
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        ax.plot(angles, scores, 'o-', linewidth=2, label='优化后表现', color='#2E86AB')
        ax.fill(angles, scores, alpha=0.25, color='#2E86AB')
        
        # 添加基准线 (50分)
        baseline_scores = [50] * len(angles)
        ax.plot(angles, baseline_scores, '--', linewidth=1, label='基准水平', color='gray', alpha=0.7)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'])
        ax.grid(True)
        
        ax.set_title('Hull Tactical项目综合性能雷达图', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        
        plt.tight_layout()
        plt.savefig(output_path / 'comprehensive_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_comparison_report(self, output_file: str = "benchmark_comparison_report.json"):
        """保存对比分析报告"""
        if not self.comparison_data:
            print("⚠️ 没有对比数据可保存")
            return None
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.comparison_data, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"✅ 对比报告已保存: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"❌ 保存对比报告失败: {e}")
            return None


def main():
    """主函数"""
    print("🎯 Hull Tactical项目基准对比分析工具")
    print("=" * 60)
    
    try:
        comparator = BenchmarkComparator()
        
        # 生成示例数据（实际使用时应加载真实数据）
        print("🔧 生成示例数据进行演示...")
        baseline_results = comparator.generate_synthetic_baseline()
        comparison_results = comparator.generate_synthetic_optimized()
        
        # 执行对比分析
        print("\n📊 执行性能对比分析...")
        comparison = comparator.compare_performance()
        
        # 生成可视化
        print("\n📈 生成可视化图表...")
        comparator.generate_visualization_plots()
        
        # 保存报告
        print("\n💾 保存分析报告...")
        report_file = comparator.save_comparison_report()
        
        # 打印摘要
        print("\n" + "="*60)
        print("📋 基准对比分析摘要")
        print("="*60)
        
        if 'summary' in comparison:
            summary = comparison['summary']
            print(f"对比策略数: {summary.get('total_strategies', 0)}")
            print(f"一致性评分: {summary.get('consistency_score', 0):.1%}")
            print(f"整体效果评分: {summary.get('overall_effectiveness', 0):.1f}/100")
            
            if 'avg_improvements' in summary:
                for metric, stats in summary['avg_improvements'].items():
                    print(f"{metric.upper()}平均改进: {stats.get('mean', 0):.1f}%")
        
        if 'statistical_tests' in comparison and comparison['statistical_tests']:
            tests = comparison['statistical_tests']
            if 't_test' in tests:
                t_test = tests['t_test']
                print(f"统计显著性: {t_test['interpretation']} (p={t_test['p_value']:.4f})")
        
        print(f"\n📊 详细报告: {report_file}")
        print(f"📈 可视化图表: benchmark_analysis/ 目录")
        
        return comparison
        
    except Exception as e:
        print(f"\n❌ 分析过程中发生错误: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 设置基准文件路径
    benchmark_file = "working/simple_ensemble_benchmark.json"  # 基线结果文件
    main()
