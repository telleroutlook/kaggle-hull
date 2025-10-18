#!/usr/bin/env python
"""
Kaggle Notebook运行脚本 - 优化版
这个脚本旨在在Kaggle环境中运行Hull Tactical模型
同时也兼容本地环境运行
"""

import sys
import os
import subprocess
import time
from pathlib import Path

def detect_environment():
    """检测运行环境"""
    if '/kaggle/input/' in os.getcwd() or '/kaggle/working' in os.getcwd():
        return 'kaggle'
    elif os.path.exists('input') and os.path.exists('working'):
        return 'local'
    else:
        return 'unknown'

# 添加项目路径 - 支持不同的数据集名称和环境
def setup_paths():
    """设置正确的项目路径"""
    
    env = detect_environment()
    print(f"🔍 检测到运行环境: {env}")
    
    # 首先检查当前目录结构
    print(f"当前工作目录: {os.getcwd()}")
    print("当前目录内容:")
    for item in os.listdir('.'):
        print(f"  - {item}")
    
    if env == 'kaggle':
        # Kaggle环境路径处理
        print("/kaggle/input目录内容:")
        for root, dirs, files in os.walk('/kaggle/input', topdown=True):
            # 只显示前两层目录
            level = root.replace('/kaggle/input', '').count(os.sep)
            if level < 3:  # 增加一层深度
                print(f"  {'  ' * level}- {os.path.basename(root)}/")
                for d in dirs[:5]:  # 限制显示数量
                    if level < 2:
                        print(f"    {'  ' * (level+1)}- {d}/")
                for f in files[:5]:  # 限制显示数量
                    if level < 2:
                        print(f"    {'  ' * (level+1)}- {f}")
            if level > 3:
                del dirs[:]  # 不再深入遍历
        
        # 查找包含working目录的数据集 - 更广泛的搜索
        found_paths = []
        if os.path.exists('/kaggle/input'):
            for root, dirs, files in os.walk('/kaggle/input'):
                if 'working' in dirs and ('main.py' in os.listdir(os.path.join(root, 'working')) or 
                                          os.path.exists(os.path.join(root, 'working', 'main.py'))):
                    working_path = os.path.join(root, 'working')
                    dataset_root = root
                    found_paths.append((working_path, dataset_root))
                    print(f"✅ 找到工作目录: {working_path}")
                    print(f"   数据集根目录: {dataset_root}")
        
        # 特别检查常见的路径
        common_paths = [
            '/kaggle/input/hullsolver/working',
            '/kaggle/input/hull-tactical-market-prediction/working',
            '/kaggle/input/hull-solver/working'
        ]
        
        for specific_path in common_paths:
            if os.path.exists(specific_path):
                print(f"✅ 找到特定路径: {specific_path}")
                dataset_root = os.path.dirname(specific_path)
                found_paths.append((specific_path, dataset_root))
        
        # 如果找到，添加第一个找到的路径
        if found_paths:
            working_path, dataset_root = found_paths[0]
            sys.path.insert(0, working_path)
            print(f"✅ 添加项目路径: {working_path}")
            return dataset_root
            
    elif env == 'local':
        # 本地环境路径处理
        local_paths = [
            'working',
            '../working',
            './working'
        ]
        
        for path in local_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, 'main.py')):
                sys.path.insert(0, path)
                print(f"✅ 找到本地工作目录: {path}")
                return '.'
    
    # 检查当前目录下是否有working
    if os.path.exists('working') and os.path.exists('working/main.py'):
        sys.path.insert(0, 'working')
        print("✅ 使用当前目录下的working文件夹")
        return '.'
    
    # 最后尝试直接在当前目录查找文件
    if os.path.exists('main.py'):
        sys.path.insert(0, '.')
        print("✅ 使用当前目录作为项目路径")
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
    
    # 根据数据集根目录确定requirements路径
    if dataset_root == '.':
        requirements_path = 'requirements.txt'
    else:
        requirements_path = os.path.join(dataset_root, 'requirements.txt')
    
    print(f"查找requirements文件: {requirements_path}")
    print(f"文件是否存在: {os.path.exists(requirements_path)}")
    
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
        import main
        
        # 保存原始的sys.argv
        original_argv = sys.argv[:]
        
        # 设置空的参数列表，避免Jupyter内核传递的参数干扰
        sys.argv = [sys.argv[0]]  # 只保留脚本名称
        
        # 运行主函数
        start_time = time.time()
        result = main.main()
        end_time = time.time()
        
        # 恢复原始的sys.argv
        sys.argv = original_argv
        
        print(f"✅ 模型运行完成，耗时: {end_time - start_time:.2f}秒")
        return result
        
    except Exception as e:
        print(f"❌ 运行模型时出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """主函数"""
    
    print("🎯 Hull Tactical - Market Prediction 模型运行器")
    print("="*50)
    
    if not dataset_root:
        print("❌ 错误: 未找到有效的数据集路径")
        return 1
    
    env = detect_environment()
    print(f"🏠 运行环境: {env}")
    
    # 安装依赖
    install_dependencies()
    
    # 检查必要的文件
    required_files = [
        'main.py',
        'lib/models.py',
        'lib/features.py',
        'lib/utils.py',
        'config.ini'
    ]
    
    # 根据数据集根目录调整文件路径检查方式
    all_files_found = True
    for req_file in required_files:
        # 构建完整路径
        if dataset_root == '.':
            full_path = req_file
        else:
            full_path = os.path.join(dataset_root, 'working', req_file)
        
        if os.path.exists(full_path):
            print(f"✅ 找到文件: {full_path}")
        else:
            # 尝试另一种路径结构
            alt_path = os.path.join(dataset_root, req_file)
            if os.path.exists(alt_path):
                print(f"✅ 找到文件: {alt_path}")
            else:
                print(f"❌ 缺少必需文件: {full_path}")
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
        '/kaggle/working/hull_metrics.csv',
        './submission.csv',
        './hull_logs.jsonl',
        './hull_metrics.csv'
    ]
    
    for output_file in output_files:
        if os.path.exists(output_file):
            print(f"✅ 输出文件已创建: {output_file}")
        else:
            print(f"⚠️ 输出文件未找到: {output_file}")
    
    return result


if __name__ == "__main__":
    sys.exit(main())