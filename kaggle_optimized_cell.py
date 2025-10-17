#!/usr/bin/env python
"""
Kaggle Notebook运行脚本 - 优化版
这个脚本旨在在Kaggle环境中运行Hull Tactical模型
"""

import sys
import os
import subprocess
import time
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/kaggle/input/hull-tactical-market-prediction/working')

def install_dependencies():
    """安装依赖包"""
    
    requirements_path = '/kaggle/input/hull-tactical-market-prediction/requirements.txt'
    
    if os.path.exists(requirements_path):
        print("📦 安装依赖包...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '-r', requirements_path
            ])
            print("✅ 依赖包安装完成")
        except Exception as e:
            print(f"⚠️ 安装依赖包时出错: {e}")
            # 尝试安装关键依赖
            essential_packages = ['numpy', 'pandas', 'scikit-learn', 'psutil']
            for package in essential_packages:
                try:
                    __import__(package)
                except ImportError:
                    print(f"📦 安装关键包: {package}")
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    else:
        print("⚠️ 未找到requirements.txt，安装关键依赖包")
        packages_to_install = ['numpy', 'pandas', 'scikit-learn', 'psutil']
        for package in packages_to_install:
            try:
                __import__(package)
            except ImportError:
                print(f"📦 安装 {package}")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])


def run_model():
    """运行模型"""
    
    print("🚀 启动Hull Tactical - Market Prediction模型")
    
    try:
        # 导入主模块
        from main import main
        
        # 运行主函数
        start_time = time.time()
        result = main()
        end_time = time.time()
        
        print(f"✅ 模型运行完成，耗时: {end_time - start_time:.2f}秒")
        return result
        
    except Exception as e:
        print(f"❌ 运行模型时出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """主函数"""
    
    print("🎯 Kaggle环境模型部署优化版")
    print("="*50)
    
    # 检查当前环境
    if '/kaggle/input/' not in os.getcwd() and '/kaggle/working' not in os.getcwd():
        print("⚠️ 警告: 似乎不在Kaggle环境中运行")
    
    # 安装依赖
    install_dependencies()
    
    # 检查必要的文件
    required_files = [
        '/kaggle/input/hull-tactical-market-prediction/working/main.py',
        '/kaggle/input/hull-tactical-market-prediction/working/lib/models.py',
        '/kaggle/input/hull-tactical-market-prediction/working/lib/features.py',
        '/kaggle/input/hull-tactical-market-prediction/working/lib/utils.py',
        '/kaggle/input/hull-tactical-market-prediction/working/config.ini'
    ]
    
    for req_file in required_files:
        if not os.path.exists(req_file):
            print(f"❌ 缺少必需文件: {req_file}")
            return 1
        else:
            print(f"✅ 找到文件: {req_file}")
    
    print("\n📋 运行模型...")
    result = run_model()
    
    # 检查输出文件
    output_files = [
        '/kaggle/working/submission.csv',
        '/kaggle/working/hull_logs.jsonl',
        '/kaggle/working/hull_metrics.csv'
    ]
    
    for output_file in output_files:
        if os.path.exists(output_file):
            print(f"✅ 输出文件已创建: {output_file}")
        else:
            print(f"⚠️ 输出文件未找到: {output_file}")
    
    return result


if __name__ == "__main__":
    sys.exit(main())