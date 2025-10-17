"""
简化测试 - 验证基本功能
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """测试所有模块导入"""
    
    modules_to_test = [
        'lib.env',
        'lib.data', 
        'lib.features',
        'lib.models',
        'lib.utils',
        'lib.evaluation'
    ]
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ {module_name} 导入成功")
        except ImportError as e:
            print(f"❌ {module_name} 导入失败: {e}")
            return False
    
    return True

def test_basic_functionality():
    """测试基本功能"""
    
    try:
        from lib.env import detect_run_environment, get_data_paths
        
        # 测试环境检测
        env = detect_run_environment()
        print(f"✅ 环境检测: {env}")
        
        # 测试路径获取
        paths = get_data_paths(env)
        print(f"✅ 路径获取: {paths}")
        
        # 测试模型创建
        from lib.models import HullModel
        model = HullModel(model_type="baseline")
        print(f"✅ 模型创建: {model.model_type}")
        
        # 测试工具函数
        from lib.utils import PerformanceTracker
        tracker = PerformanceTracker()
        tracker.start_task("test")
        tracker.end_task()
        print(f"✅ 性能跟踪器工作正常")
        
        return True
        
    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    
    print("🚀 开始简化测试")
    print("="*50)
    
    success = True
    
    print("\n📋 测试模块导入...")
    if not test_imports():
        success = False
    
    print("\n📋 测试基本功能...")
    if not test_basic_functionality():
        success = False
    
    print("\n" + "="*50)
    if success:
        print("🎉 所有简化测试通过！")
    else:
        print("❌ 部分测试失败")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())