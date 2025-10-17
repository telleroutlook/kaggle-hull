#!/usr/bin/env python3
"""
简化测试版本
"""

import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(__file__))

try:
    from lib.env import detect_run_environment, get_data_paths
    print("✅ 模块导入成功")
    
    # 测试环境检测
    env = detect_run_environment()
    print(f"环境: {env}")
    
    # 测试路径获取
    data_paths = get_data_paths(env)
    print(f"数据路径: {data_paths.test_data}")
    
    print("✅ 测试完成")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()