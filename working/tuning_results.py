#!/usr/bin/env python3
"""
Hull Tactical - 超参数调优结果分析和报告系统
生成详细的性能分析报告、对比图表和优化建议
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import warnings
from dataclasses import asdict
import pickle

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly未安装，将使用matplotlib作为后备")


class TuningResultAnalyzer:
    """调优结果分析器"""
    
    def __init__(self, results_dir: str = "tuning_results"):
        self.results_dir = Path(results_dir)
        self.results_data = {}
        self.summary_stats = {}
        
    def load_results(self, file_pattern: str = "*.json") -> Dict[str, Any]:
        """加载调优结果文件"""
        result_files = list(self.results_dir.glob(file_pattern))
        
        if not result_files:
            print(f"在 {self.results_dir} 中未找到结果文件")
            return {}
        
        all_results = {}
        for file_path in result_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # 提取基本信息
                file_key = file_path.stem
                all_results[file_key] = {
                    "file_path": str(file_path),
                    "config": data.get("config", {}),
                    "results": data.get("results", {}),
                    "rankings": data.get("rankings", []),
                    "timestamp": data.get("timestamp", "")
                }
                
            except Exception as e:
                print(f"加载文件 {file_path} 失败: {e}")
                continue
        
        self.results_data = all_results
        return all_results
    
    def analyze_model_performance(self) -> pd.DataFrame:
        """分析模型性能"""
        performance_data = []
        
        for file_key, data in self.results_data.items():
            results = data.get("results", {})
            
            for model_type, result in results.items():
                # 提取CV分数
                cv_scores = result.get("cv_scores", {})
                mean_scores = result.get("mean_scores", {})
                std_scores = result.get("std_scores", {})
                
                row = {
                    "file_key": file_key,
                    "model_type": model_type,
                    "best_score": result.get("best_score", np.nan),
                    "tuning_time": result.get("tuning_time", np.nan),
                    "n_trials": result.get("n_trials", 0),
                    "config": data.get("config", {})
                }
                
                # 添加各个指标的均值和标准差
                for metric in ["mse", "mae", "r2"]:
                    if metric in mean_scores:
                        row[f"{metric}_mean"] = mean_scores[metric]
                        row[f"{metric}_std"] = std_scores.get(metric, np.nan)
                    else:
                        row[f"{metric}_mean"] = np.nan
                        row[f"{metric}_std"] = np.nan
                
                performance_data.append(row)
        
        self.performance_df = pd.DataFrame(performance_data)
        return self.performance_df
    
    def generate_performance_summary(self) -> Dict[str, Any]:
        """生成性能摘要"""
        if not hasattr(self, 'performance_df'):
            self.analyze_model_performance()
        
        summary = {
            "total_experiments": len(self.performance_df),
            "unique_models": self.performance_df['model_type'].nunique(),
            "best_overall": {},
            "model_rankings": {},
            "performance_improvements": {},
            "statistical_summary": {}
        }
        
        # 最佳模型
        best_overall_idx = self.performance_df['mse_mean'].idxmin()
        best_overall = self.performance_df.loc[best_overall_idx]
        summary["best_overall"] = {
            "model_type": best_overall['model_type'],
            "mse": best_overall['mse_mean'],
            "file_key": best_overall['file_key'],
            "improvement_vs_baseline": self._calculate_improvement(best_overall)
        }
        
        # 模型排名
        model_rankings = (self.performance_df.groupby('model_type')['mse_mean']
                         .mean()
                         .sort_values()
                         .to_dict())
        summary["model_rankings"] = model_rankings
        
        # 性能改进（如果有基线模型）
        if 'random_forest' in model_rankings:
            baseline_mse = model_rankings['random_forest']
            improvements = {}
            for model, mse in model_rankings.items():
                if model != 'random_forest':
                    improvement = (baseline_mse - mse) / baseline_mse * 100
                    improvements[model] = f"{improvement:.2f}%"
            summary["performance_improvements"] = improvements
        
        # 统计摘要
        summary["statistical_summary"] = self.performance_df.groupby('model_type').agg({
            'mse_mean': ['mean', 'std', 'min', 'max'],
            'mae_mean': ['mean', 'std', 'min', 'max'],
            'r2_mean': ['mean', 'std', 'min', 'max'],
            'tuning_time': ['mean', 'std', 'min', 'max']
        }).round(6).to_dict()
        
        self.summary_stats = summary
        return summary
    
    def _calculate_improvement(self, result_row) -> str:
        """计算性能改进"""
        # 这里可以与基线模型比较
        baseline_mse = 0.25  # 假设基线MSE
        improvement = (baseline_mse - result_row['mse_mean']) / baseline_mse * 100
        return f"{improvement:.2f}%"
    
    def create_performance_comparison_chart(self, output_path: str = None) -> str:
        """创建性能对比图表"""
        if not hasattr(self, 'performance_df'):
            self.analyze_model_performance()
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Hull Tactical - 超参数调优性能分析', fontsize=16, fontweight='bold')
        
        # 1. MSE对比
        ax1 = axes[0, 0]
        mse_data = self.performance_df.dropna(subset=['mse_mean'])
        if not mse_data.empty:
            mse_data.boxplot(column='mse_mean', by='model_type', ax=ax1)
            ax1.set_title('MSE分布对比')
            ax1.set_xlabel('模型类型')
            ax1.set_ylabel('MSE')
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. MAE对比
        ax2 = axes[0, 1]
        mae_data = self.performance_df.dropna(subset=['mae_mean'])
        if not mae_data.empty:
            mae_data.boxplot(column='mae_mean', by='model_type', ax=ax2)
            ax2.set_title('MAE分布对比')
            ax2.set_xlabel('模型类型')
            ax2.set_ylabel('MAE')
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. R²对比
        ax3 = axes[1, 0]
        r2_data = self.performance_df.dropna(subset=['r2_mean'])
        if not r2_data.empty:
            r2_data.boxplot(column='r2_mean', by='model_type', ax=ax3)
            ax3.set_title('R²分布对比')
            ax3.set_xlabel('模型类型')
            ax3.set_ylabel('R²')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. 调优时间对比
        ax4 = axes[1, 1]
        time_data = self.performance_df.dropna(subset=['tuning_time'])
        if not time_data.empty:
            time_data.boxplot(column='tuning_time', by='model_type', ax=ax4)
            ax4.set_title('调优时间对比')
            ax4.set_xlabel('模型类型')
            ax4.set_ylabel('调优时间(秒)')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if output_path is None:
            output_path = self.results_dir / "performance_comparison.png"
        else:
            output_path = Path(output_path)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def create_model_ranking_chart(self, output_path: str = None) -> str:
        """创建模型排名图表"""
        if not self.summary_stats:
            self.generate_performance_summary()
        
        model_rankings = self.summary_stats['model_rankings']
        
        # 创建排名图表
        models = list(model_rankings.keys())
        scores = list(model_rankings.values())
        
        # 按分数排序
        sorted_data = sorted(zip(models, scores), key=lambda x: x[1])
        models_sorted, scores_sorted = zip(*sorted_data)
        
        plt.figure(figsize=(12, 6))
        
        # 创建条形图
        bars = plt.bar(models_sorted, scores_sorted, 
                      color=['#2E8B57', '#4682B4', '#CD853F', '#DC143C'][:len(models_sorted)])
        
        # 添加数值标签
        for bar, score in zip(bars, scores_sorted):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Hull Tactical - 模型性能排名 (MSE)', fontsize=14, fontweight='bold')
        plt.xlabel('模型类型', fontsize=12)
        plt.ylabel('MSE (越低越好)', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # 添加性能改进标注
        if 'random_forest' in model_rankings:
            baseline_score = model_rankings['random_forest']
            for i, (model, score) in enumerate(zip(models_sorted, scores_sorted)):
                if model != 'random_forest':
                    improvement = (baseline_score - score) / baseline_score * 100
                    plt.text(i, score/2, f'+{improvement:.1f}%', 
                            ha='center', va='center', fontsize=10, 
                            color='white', fontweight='bold')
        
        plt.tight_layout()
        
        if output_path is None:
            output_path = self.results_dir / "model_ranking.png"
        else:
            output_path = Path(output_path)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def analyze_parameter_importance(self) -> Dict[str, Any]:
        """分析参数重要性"""
        param_importance = {}
        
        for file_key, data in self.results_data.items():
            results = data.get("results", {})
            
            for model_type, result in results.items():
                best_params = result.get("best_params", {})
                cv_scores = result.get("cv_scores", {})
                
                if not best_params:
                    continue
                
                # 计算每个参数的变异系数
                param_stats = {}
                for param, value in best_params.items():
                    if isinstance(value, (int, float)):
                        # 对于数值参数，记录其值
                        param_stats[param] = {
                            "value": value,
                            "type": "numeric"
                        }
                    else:
                        # 对于分类参数，记录其值
                        param_stats[param] = {
                            "value": str(value),
                            "type": "categorical"
                        }
                
                param_importance[f"{file_key}_{model_type}"] = {
                    "best_params": param_stats,
                    "performance": {
                        "mse_mean": np.mean(cv_scores.get("mse", [np.nan])),
                        "mse_std": np.std(cv_scores.get("mse", [np.nan])),
                        "n_trials": result.get("n_trials", 0)
                    }
                }
        
        return param_importance
    
    def generate_recommendations(self) -> List[str]:
        """生成优化建议"""
        if not self.summary_stats:
            self.generate_performance_summary()
        
        recommendations = []
        
        # 基于性能分析的建议
        model_rankings = self.summary_stats['model_rankings']
        
        if model_rankings:
            best_model = min(model_rankings.keys(), key=lambda x: model_rankings[x])
            worst_model = max(model_rankings.keys(), key=lambda x: model_rankings[x])
            
            recommendations.append(f"🏆 最佳模型: {best_model} (MSE: {model_rankings[best_model]:.4f})")
            recommendations.append(f"📊 建议重点优化: {worst_model} (当前MSE: {model_rankings[worst_model]:.4f})")
            
            # 性能差异分析
            if len(model_rankings) >= 2:
                scores = list(model_rankings.values())
                score_diff = max(scores) - min(scores)
                if score_diff > 0.01:  # 如果差异显著
                    recommendations.append(f"⚠️ 模型间性能差异较大 ({score_diff:.4f})，建议进行集成学习")
            
            # 调优效率建议
            perf_df = self.performance_df
            if not perf_df.empty:
                avg_tuning_time = perf_df['tuning_time'].mean()
                if avg_tuning_time > 600:  # 如果调优时间超过10分钟
                    recommendations.append(f"⏱️ 平均调优时间较长 ({avg_tuning_time:.0f}s)，建议减少试验次数或使用并行计算")
                
                # 检查过拟合风险
                for model in model_rankings.keys():
                    model_data = perf_df[perf_df['model_type'] == model]
                    if not model_data.empty:
                        mse_std = model_data['mse_std'].mean()
                        if not np.isnan(mse_std) and mse_std > 0.1:
                            recommendations.append(f"🔍 {model}存在过拟合风险，CV标准差较大 ({mse_std:.4f})，建议增加正则化")
        
        # 基于参数分析的建议
        param_importance = self.analyze_parameter_importance()
        if param_importance:
            recommendations.append("📈 参数分析完成，建议查看详细参数重要性报告")
        
        return recommendations
    
    def generate_html_report(self, output_path: str = None) -> str:
        """生成HTML报告"""
        if output_path is None:
            output_path = self.results_dir / "tuning_report.html"
        else:
            output_path = Path(output_path)
        
        # 生成分析数据
        performance_df = self.analyze_model_performance()
        summary = self.generate_performance_summary()
        recommendations = self.generate_recommendations()
        
        # 创建图表
        chart1_path = self.create_performance_comparison_chart()
        chart2_path = self.create_model_ranking_chart()
        
        # HTML模板
        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Hull Tactical - 超参数调优报告</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    text-align: center;
                    margin-bottom: 30px;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495e;
                    margin-top: 30px;
                    border-left: 4px solid #3498db;
                    padding-left: 15px;
                }}
                .summary-box {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin: 20px 0;
                }}
                .metric-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .metric-card {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    border-left: 4px solid #3498db;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .metric-label {{
                    font-size: 14px;
                    color: #7f8c8d;
                    margin-top: 5px;
                }}
                .chart-container {{
                    text-align: center;
                    margin: 30px 0;
                }}
                .chart-container img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }}
                .recommendations {{
                    background: #e8f5e8;
                    border: 1px solid #4caf50;
                    border-radius: 8px;
                    padding: 20px;
                    margin: 20px 0;
                }}
                .recommendations h3 {{
                    color: #2e7d32;
                    margin-top: 0;
                }}
                .recommendations ul {{
                    list-style-type: none;
                    padding: 0;
                }}
                .recommendations li {{
                    margin: 10px 0;
                    padding: 10px;
                    background: white;
                    border-radius: 5px;
                    border-left: 4px solid #4caf50;
                }}
                .data-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                .data-table th, .data-table td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                .data-table th {{
                    background-color: #f2f2f2;
                    font-weight: bold;
                }}
                .data-table tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 50px;
                    color: #7f8c8d;
                    font-size: 14px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 Hull Tactical - 超参数调优报告</h1>
                
                <div class="summary-box">
                    <h2>📊 调优摘要</h2>
                    <p><strong>生成时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>实验数量:</strong> {summary['total_experiments']}</p>
                    <p><strong>模型类型:</strong> {summary['unique_models']}</p>
                    <p><strong>最佳模型:</strong> {summary['best_overall'].get('model_type', 'N/A')}</p>
                </div>
                
                <h2>🏆 性能排名</h2>
                <div class="metric-grid">
                    {self._generate_ranking_cards(summary['model_rankings'])}
                </div>
                
                <h2>📈 性能分析图表</h2>
                <div class="chart-container">
                    <h3>模型性能对比</h3>
                    <img src="{Path(chart1_path).name}" alt="性能对比图">
                </div>
                
                <div class="chart-container">
                    <h3>模型排名</h3>
                    <img src="{Path(chart2_path).name}" alt="模型排名图">
                </div>
                
                <h2>💡 优化建议</h2>
                <div class="recommendations">
                    <h3>智能建议</h3>
                    <ul>
                        {self._generate_recommendations_html(recommendations)}
                    </ul>
                </div>
                
                <h2>📋 详细性能数据</h2>
                {self._generate_data_table(performance_df)}
                
                <div class="footer">
                    <p>报告由 Hull Tactical 自动生成 | 数据时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(output_path)
    
    def _generate_ranking_cards(self, rankings: Dict[str, float]) -> str:
        """生成排名卡片HTML"""
        cards = ""
        for i, (model, score) in enumerate(sorted(rankings.items(), key=lambda x: x[1]), 1):
            cards += f"""
            <div class="metric-card">
                <div class="metric-value">#{i} {model.title()}</div>
                <div class="metric-label">MSE: {score:.4f}</div>
            </div>
            """
        return cards
    
    def _generate_recommendations_html(self, recommendations: List[str]) -> str:
        """生成建议HTML"""
        items = ""
        for rec in recommendations:
            items += f"<li>{rec}</li>"
        return items
    
    def _generate_data_table(self, df: pd.DataFrame) -> str:
        """生成数据表格HTML"""
        if df.empty:
            return "<p>暂无数据</p>"
        
        # 选择关键列
        key_columns = ['model_type', 'mse_mean', 'mae_mean', 'r2_mean', 'tuning_time', 'n_trials']
        display_df = df[key_columns].round(4)
        
        table_html = "<table class='data-table'><tr>"
        for col in display_df.columns:
            table_html += f"<th>{col.title()}</th>"
        table_html += "</tr>"
        
        for _, row in display_df.iterrows():
            table_html += "<tr>"
            for col in display_df.columns:
                value = row[col]
                if pd.isna(value):
                    value = "N/A"
                table_html += f"<td>{value}</td>"
            table_html += "</tr>"
        
        table_html += "</table>"
        return table_html


def main():
    """主函数 - 生成完整的调优报告"""
    analyzer = TuningResultAnalyzer()
    
    # 加载结果
    print("🔍 正在加载调优结果...")
    results = analyzer.load_results()
    
    if not results:
        print("❌ 未找到调优结果文件")
        return
    
    print(f"✅ 成功加载 {len(results)} 个实验结果")
    
    # 生成分析
    print("📊 正在生成性能分析...")
    analyzer.analyze_model_performance()
    
    print("📈 正在生成性能摘要...")
    summary = analyzer.generate_performance_summary()
    
    # 生成图表
    print("🎨 正在生成图表...")
    chart1 = analyzer.create_performance_comparison_chart()
    chart2 = analyzer.create_model_ranking_chart()
    
    # 生成建议
    print("💡 正在生成优化建议...")
    recommendations = analyzer.generate_recommendations()
    
    # 生成HTML报告
    print("📝 正在生成HTML报告...")
    html_report = analyzer.generate_html_report()
    
    # 输出结果
    print("\n" + "="*60)
    print("🎉 调优结果分析完成！")
    print("="*60)
    
    print(f"\n📁 输出文件:")
    print(f"   📊 性能对比图: {chart1}")
    print(f"   🏆 模型排名图: {chart2}")
    print(f"   📄 HTML报告: {html_report}")
    
    print(f"\n🏆 最佳模型: {summary['best_overall'].get('model_type', 'N/A')}")
    print(f"   MSE: {summary['best_overall'].get('mse', 'N/A')}")
    
    print(f"\n💡 优化建议:")
    for rec in recommendations[:5]:  # 显示前5个建议
        print(f"   {rec}")
    
    print(f"\n📋 完整报告请查看: {html_report}")


if __name__ == "__main__":
    main()
