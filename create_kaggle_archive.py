#!/usr/bin/env python3
"""
为Hull Tactical - Market Prediction创建Kaggle部署归档的脚本
这会创建一个压缩归档文件，可以上传到Kaggle运行模型
"""

import os
import shutil
import zipfile

def create_kaggle_archive():
    """创建包含Kaggle部署所需所有文件的zip归档"""
    
    # 要包含在归档中的文件和目录
    files_to_include = [
        # 主模型文件
        "working/main.py",
        "working/__init__.py",
        "working/config.ini",
        
        # 核心库文件
        "working/lib/",
        
        # 测试文件
        "working/tests/",
        
        # 依赖文件
        "requirements.txt",
        
        # 文档
        "README.md",
        "IFLOW.md",
        "KAGGLE_DEPLOYMENT.md",
        
        # Kaggle脚本
        "kaggle_simple_cell.py",
        "create_kaggle_archive.py",
    ]
    
    # 创建输出目录
    os.makedirs("input", exist_ok=True)
    
    # 创建归档
    archive_path = "input/kaggle_hull_solver.zip"
    
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        added_files = set()
        
        for item in files_to_include:
            if os.path.isdir(item):
                # 递归添加目录
                for root, dirs, files in os.walk(item):
                    for file in files:
                        # 跳过缓存和临时文件
                        if '__pycache__' in root or file.endswith('.pyc') or file.endswith('.pyo'):
                            continue
                        file_path = os.path.join(root, file)
                        # 在归档中使用相对路径，但保留working/结构
                        arcname = os.path.relpath(file_path, '.')  # 保留完整相对路径
                        if arcname not in added_files:
                            zipf.write(file_path, arcname)
                            added_files.add(arcname)
                            print(f"Added: {arcname}")
            else:
                # 添加单个文件
                if os.path.exists(item):
                    # 在归档中保留working/结构
                    arcname = item  # 使用完整相对路径
                    if arcname not in added_files:
                        zipf.write(item, arcname)
                        added_files.add(arcname)
                        print(f"Added: {arcname}")
                else:
                    print(f"Warning: {item} not found, skipping")
    
    print(f"\n✅ Created Kaggle deployment archive: {archive_path}")
    print(f"📦 Archive size: {os.path.getsize(archive_path) / (1024*1024):.2f} MB")
    print(f"📁 Total files added: {len(added_files)}")

if __name__ == "__main__":
    create_kaggle_archive()