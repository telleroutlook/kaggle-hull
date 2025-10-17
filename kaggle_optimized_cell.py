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

# 添加项目路径 - 支持不同的数据集名称
def setup_paths():
    """设置正确的项目路径"""
    
    # 可能的路径模式
    possible_paths = [
        '/kaggle/input/hullsolver/working',
        '/kaggle/input/hull-tactical-market-prediction/working',
        '/kaggle/input/kaggle-hull-solver/working',
        '/kaggle/input/hull-solver/working'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            sys.path.insert(0, path)
            print(f"✅ 找到并添加路径: {path}")
            return path.replace('/working', '')  # 返回数据集根路径
    
    # 如果都没找到，尝试当前目录下的working
    if os.path.exists('working'):
        sys.path.insert(0, 'working')
        print("✅ 使用当前目录下的working文件夹")
        return '.'
    
    print("❌ 未找到有效的项目路径")
    return None

# 设置路径
dataset_root = setup_paths()

def install_dependencies():
    """安装依赖包"""
    
    if not dataset_root:
        print("❌ 无法安装依赖：未找到数据集路径")
        return
    
    requirements_path = os.path.join(dataset_root, 'requirements.txt')
    
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
    
    if not dataset_root:
        print("❌ 错误: 未找到有效的数据集路径")
        return 1
    
    # 检查当前环境
    if '/kaggle/input/' not in os.getcwd() and '/kaggle/working' not in os.getcwd():
        print("⚠️ 警告: 似乎不在Kaggle环境中运行")
    
    # 安装依赖
    install_dependencies()
    
    # 检查必要的文件
    required_files = [
        os.path.join(dataset_root, 'working/main.py'),
        os.path.join(dataset_root, 'working/lib/models.py'),
        os.path.join(dataset_root, 'working/lib/features.py'),
        os.path.join(dataset_root, 'working/lib/utils.py'),
        os.path.join(dataset_root, 'working/config.ini')
    ]
    
    all_files_found = True
    for req_file in required_files:
        if os.path.exists(req_file):
            print(f"✅ 找到文件: {req_file}")
        else:
            print(f"❌ 缺少必需文件: {req_file}")
            all_files_found = False
    
    if not all_files_found:
        print("❌ 一些必需文件缺失，请检查数据集是否正确上传")
        return 1
    
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