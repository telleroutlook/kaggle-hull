# Hull Tactical - Market Prediction - Kaggle Cell Version (Fixed)
# 将整个代码复制粘贴到单个Kaggle notebook cell中

import os
import sys
import json
import subprocess

TRUE_STRINGS = {"1", "true", "yes", "on"}
VERBOSE = os.getenv("VERBOSE", "0").lower() in TRUE_STRINGS


def vlog(message: str) -> None:
    if VERBOSE:
        print(message)

# 自动查找模型目录的函数
def find_solver_path():
    """通过在/kaggle/input中搜索自动查找模型路径"""
    print("正在搜索模型目录...")
    
    # Kaggle可能解压缩模型的不同路径
    possible_paths = [
        "/kaggle/input/hull01",  # 当前上传的数据集根目录
        "/kaggle/input/hull01/main.py",
        "/kaggle/input/hull01/working/main.py",
        "/kaggle/input/hull01/other/default/1",
        "/kaggle/input/hull01/other/default/1/working/main.py",
        "/kaggle/input/hull-tactical-solver",
        "/kaggle/input/hull-tactical-solver/main.py",
        "/kaggle/input/hull-tactical-solver/working/main.py",
    ]
    
    # 首先检查特定的路径是否存在
    for path in possible_paths:
        if os.path.exists(path):
            # 如果是文件路径，获取目录
            if os.path.isfile(path):
                # 如果找到main.py，检查是否在working目录中
                dir_path = os.path.dirname(path)
                if os.path.basename(dir_path) == "working":
                    # 在working目录中，需要向上一级
                    return os.path.dirname(dir_path)
                else:
                    return dir_path
            else:
                return path
    
    # 如果特定路径未找到，递归搜索/kaggle/input
    print("在/kaggle/input中递归搜索...")
    for root, dirs, files in os.walk("/kaggle/input"):
        if "main.py" in files:
            print(f"在以下位置找到main.py: {root}")
            # 检查是否在working目录中
            if os.path.basename(root) == "working":
                return os.path.dirname(root)
            else:
                return root
        # 同时检查其他模型指示文件
        if any(f.endswith('.py') and f in ['main.py', 'model.py', 'hull_model.py'] for f in files):
            print(f"找到模型文件在: {root}")
            if os.path.basename(root) == "working":
                return os.path.dirname(root)
            else:
                return root
    
    # 如果仍未找到，查找包含'solver'或'hull'的目录
    for root, dirs, files in os.walk("/kaggle/input"):
        if "solver" in root.lower() or "hull" in root.lower():
            print(f"找到相关目录: {root}")
            return root
    
    return None

# 自动查找模型路径
solver_path = find_solver_path()
if solver_path:
    print(f"使用模型基础路径: {solver_path}")
    
    # 检查是否需要添加'working'子目录
    working_path = os.path.join(solver_path, "working")
    if os.path.exists(working_path) and os.path.isdir(working_path):
        print(f"找到working目录: {working_path}")
        # 使用working目录作为主要路径
        actual_solver_path = working_path
    else:
        # 检查main.py是否直接在模型路径中
        main_py_path = os.path.join(solver_path, "main.py")
        if os.path.exists(main_py_path):
            print(f"在以下位置直接找到main.py: {solver_path}")
            actual_solver_path = solver_path
        else:
            print("❌ 在模型目录中找不到main.py")
            actual_solver_path = solver_path
else:
    print("❌ 无法自动找到模型目录")
    print("/kaggle/input中的可用目录:")
    for item in os.listdir("/kaggle/input"):
        print(f"  - {item}")
        item_path = os.path.join("/kaggle/input", item)
        if os.path.isdir(item_path):
            vlog(f"    内容: {os.listdir(item_path)}")
    # 尝试备用路径
    possible_fallbacks = [
        "/kaggle/input/hull01",
        "/kaggle/input/hull01/working",
        "/kaggle/input/hull-tactical-solver",
        "/kaggle/input/hull-tactical-solver/working",
    ]
    for fallback in possible_fallbacks:
        if os.path.exists(fallback):
            print(f"使用备用路径: {fallback}")
            actual_solver_path = fallback
            break
    else:
        print("❌ 未找到有效的备用路径")
        actual_solver_path = "/kaggle/input/hull01"

print(f"最终模型路径: {actual_solver_path}")

# 检查模型目录中的内容
print("\n检查模型目录...")
if os.path.exists(actual_solver_path):
    print(f"模型路径存在: {actual_solver_path}")
    for item in os.listdir(actual_solver_path):
        vlog(f"  - {item}")
    
    # 检查lib目录
    lib_path = os.path.join(actual_solver_path, "lib")
    if os.path.exists(lib_path):
        vlog(f"\nLib目录存在: {lib_path}")
        for item in os.listdir(lib_path):
            vlog(f"  - {item}")
    else:
        print("⚠️  Lib目录不存在，可能影响模块导入")
else:
    print(f"模型路径不存在: {actual_solver_path}")
    print("可用的输入目录:")
    for item in os.listdir("/kaggle/input"):
        print(f"  - {item}")

# 安装依赖（Kaggle notebook通常已包含这些）
print("\n检查依赖...")

def check_package(pkg_name, import_name=None):
    """Import package and print version for quick environment sanity check."""
    name = import_name or pkg_name
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "<unknown>")
        print(f"✅ {pkg_name} {version}")
        return True
    except ImportError as exc:
        print(f"❌ {pkg_name} 未找到: {exc}")
        return False

deps = [
    ("numpy", None),
    ("pandas", None),
    ("scikit-learn", "sklearn"),
    ("lightgbm", None),
    ("xgboost", None),
    ("catboost", None),
    ("pyarrow", None),
    ("psutil", None),
]

missing = [pkg for pkg, import_name in deps if not check_package(pkg, import_name)]
need_dependency_install = bool(missing)
if missing:
    print("⚠️ 上述依赖缺失，若在本地运行请先安装，Kaggle 提交需将 wheel 一起上传。")
else:
    print("环境依赖完整，可继续运行。")

# 设置数据路径
print("\n🚀 启动Hull Tactical - Market Prediction模型...")

# 切换到模型目录
original_dir = os.getcwd()
try:
    os.chdir(actual_solver_path)
    print(f"切换到模型目录: {os.getcwd()}")
    
    # 检查是否需要安装额外依赖
    # requirements.txt 可能位于working目录或其父目录
    requirements_candidates = [
        os.path.join(actual_solver_path, "requirements.txt"),
        os.path.join(os.path.dirname(actual_solver_path), "requirements.txt"),
    ]
    requirements_path = next((path for path in requirements_candidates if os.path.exists(path)), None)
    
    should_install = need_dependency_install or os.environ.get("FORCE_PIP_INSTALL") == "1"

    if requirements_path and should_install:
        print(f"安装依赖: {requirements_path}")
        pip_cmd = [sys.executable, "-m", "pip", "install", "-r", requirements_path]
        pip_result = subprocess.run(pip_cmd)
        if pip_result.returncode != 0:
            print(f"❌ pip安装失败 (exit code {pip_result.returncode})，请检查上方日志")
        else:
            print("✅ 依赖安装完成")
    elif requirements_path and not should_install:
        print("requirements.txt 已找到，但环境依赖齐全，默认跳过 pip 安装。设置 FORCE_PIP_INSTALL=1 可强制执行。")
    else:
        print("未找到requirements.txt，跳过依赖安装")
    
    # 运行推理服务器 + 网关流程
    print("运行评估API推理服务器...")
    result = subprocess.run([
        sys.executable, "inference_server.py"
    ])
    if result.returncode != 0:
        print(f"❌ 推理服务器返回非零状态 {result.returncode}，请检查上方日志")
    
    # 检查提交文件
    submission_parquet = "/kaggle/working/submission.parquet"
    submission_csv = "/kaggle/working/submission.csv"
    if os.path.exists(submission_parquet) or os.path.exists(submission_csv):
        if os.path.exists(submission_parquet):
            print(f"\n✅ 成功！生成: {submission_parquet}")
        if os.path.exists(submission_csv):
            print(f"✅ 同步生成: {submission_csv}")
        print("📊 模型运行完成")
        print("\n📁 从输出面板下载submission.parquet (或 submission.csv)")
    else:
        print("❌ 未创建submission.parquet - 请检查上面的错误")
        
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
finally:
    os.chdir(original_dir)

print("\n🏁 处理完成!")
