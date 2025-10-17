"""
运行所有测试
"""

import sys
import os
import subprocess
import argparse

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def run_pytest(test_files=None, verbose=False):
    """使用pytest运行测试"""
    
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if test_files:
        cmd.extend(test_files)
    else:
        cmd.append(".")  # 运行所有测试
    
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    return result.returncode


def run_individual_tests():
    """运行各个测试模块"""
    
    test_modules = [
        "test_env",
        "test_data", 
        "test_features",
        "test_models",
        "test_utils",
        "test_evaluation"
    ]
    
    all_passed = True
    
    for module in test_modules:
        print(f"\n{'='*50}")
        print(f"运行 {module}...")
        print('='*50)
        
        try:
            module_path = f"{module}.py"
            exec(open(os.path.join(os.path.dirname(__file__), module_path)).read())
            print(f"✅ {module} 通过")
        except Exception as e:
            print(f"❌ {module} 失败: {e}")
            all_passed = False
    
    return all_passed


def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(description="运行测试套件")
    parser.add_argument("--pytest", action="store_true", help="使用pytest运行测试")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    parser.add_argument("test_files", nargs="*", help="指定要运行的测试文件")
    
    args = parser.parse_args()
    
    print("🚀 开始运行测试套件")
    
    if args.pytest:
        # 使用pytest运行
        print("📋 使用pytest运行测试...")
        return_code = run_pytest(args.test_files, args.verbose)
        success = (return_code == 0)
    else:
        # 使用直接执行方式运行
        print("📋 使用直接执行方式运行测试...")
        success = run_individual_tests()
    
    print(f"\n{'='*50}")
    if success:
        print("🎉 所有测试通过！")
    else:
        print("❌ 部分测试失败")
    print('='*50)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())