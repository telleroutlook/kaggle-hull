# Hull Tactical - Market Prediction - Kaggle Cell Version (Fixed)
# 将整个代码复制粘贴到单个Kaggle notebook cell中

import os
import sys
import json
import subprocess
import warnings

# 配置警告处理 - 在所有其他导入之前
def configure_warnings():
    """配置警告处理以避免pandas比较警告"""
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', 
                          message='.*invalid value encountered in greater.*',
                          category=RuntimeWarning)
    warnings.filterwarnings('ignore',
                          message='.*invalid value encountered in less.*', 
                          category=RuntimeWarning)
    warnings.filterwarnings('ignore',
                          message='.*frozen modules.*',
                          category=UserWarning)
    print("✅ 警告处理已配置")

# 立即配置警告处理
configure_warnings()

TRUE_STRINGS = {"1", "true", "yes", "on"}
VERBOSE = os.getenv("VERBOSE", "0").lower() in TRUE_STRINGS


def vlog(message: str) -> None:
    if VERBOSE:
        print(message)

# 自动查找模型目录的函数
def find_solver_path():
    """通过在/kaggle/input中搜索自动查找模型路径"""
    print("正在搜索模型目录...")
    
    print("/kaggle/input中的可用目录:")
    for item in os.listdir("/kaggle/input"):
        item_path = os.path.join("/kaggle/input", item)
        if os.path.isdir(item_path):
            print(f"  - {item}/ (包含: {os.listdir(item_path)[:5]}{'...' if len(os.listdir(item_path)) > 5 else ''})")
        else:
            print(f"  - {item}")
    
    # Kaggle可能解压缩模型的不同路径 - 更全面的列表
    possible_paths = [
        "/kaggle/input/hull01",  # 常见的数据集名称
        "/kaggle/input/hull01/working",
        "/kaggle/input/hull01/main.py",
        "/kaggle/input/hull01/working/main.py",
        "/kaggle/input/hull-tactical-solver",
        "/kaggle/input/hull-tactical-solver/working",
        "/kaggle/input/hull-tactical-solver/working/main.py",
        "/kaggle/input/hull-tactical-solver/main.py",
        "/kaggle/input/kaggle_hull_solver",
        "/kaggle/input/kaggle_hull_solver/working",
        "/kaggle/input/kaggle_hull_solver/working/main.py",
        "/kaggle/input/kaggle_hull_solver/main.py",
        # 其他可能的变体
        "/kaggle/input/hull-solver",
        "/kaggle/input/hull-solver/working",
        "/kaggle/input/market-prediction-solver",
        "/kaggle/input/market-prediction-solver/working",
    ]
    
    # 首先检查特定的路径是否存在
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ 找到路径: {path}")
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
                # 如果是目录，检查是否包含inference_server.py
                inference_path = os.path.join(path, "inference_server.py")
                main_path = os.path.join(path, "main.py")
                if os.path.exists(inference_path) or os.path.exists(main_path):
                    return path
                else:
                    # 检查是否包含working子目录
                    working_path = os.path.join(path, "working")
                    if os.path.exists(working_path):
                        return path
    
    # 如果特定路径未找到，递归搜索/kaggle/input
    print("在/kaggle/input中递归搜索...")
    found_paths = []
    for root, dirs, files in os.walk("/kaggle/input"):
        # 检查关键文件
        if "inference_server.py" in files:
            print(f"✅ 找到inference_server.py在: {root}")
            found_paths.append(root)
        elif "main.py" in files and any(f.endswith('.py') for f in files):
            print(f"📄 找到main.py和Python文件在: {root}")
            found_paths.append(root)
    
    # 如果找到多个路径，优先选择包含inference_server.py的
    for path in found_paths:
        if "inference_server.py" in os.listdir(path):
            # 检查是否在working目录中
            if os.path.basename(path) == "working":
                return os.path.dirname(path)
            else:
                return path
    
    # 如果仍有多个路径，选择第一个
    if found_paths:
        path = found_paths[0]
        if os.path.basename(path) == "working":
            return os.path.dirname(path)
        else:
            return path
    
    # 最后的备用策略：查找任何包含'working'的目录
    for root, dirs, files in os.walk("/kaggle/input"):
        if "working" in dirs:
            working_path = os.path.join(root, "working")
            inference_path = os.path.join(working_path, "inference_server.py")
            if os.path.exists(inference_path):
                print(f"🔍 在working目录中找到inference_server.py: {root}")
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
    print(f"原始工作目录: {original_dir}")
    print(f"目标模型目录: {actual_solver_path}")
    
    # 确保模型目录存在且可访问
    if not os.path.exists(actual_solver_path):
        print(f"❌ 模型目录不存在: {actual_solver_path}")
        raise FileNotFoundError(f"模型目录不存在: {actual_solver_path}")
    
    os.chdir(actual_solver_path)
    print(f"切换到模型目录: {os.getcwd()}")
    
    # 检查关键文件是否存在
    inference_server_path = os.path.join(actual_solver_path, "inference_server.py")
    main_path = os.path.join(actual_solver_path, "main.py")
    
    if not os.path.exists(inference_server_path):
        print(f"❌ 关键文件不存在: {inference_server_path}")
        print("在模型目录中查找文件:")
        for item in os.listdir(actual_solver_path):
            print(f"  - {item}")
        raise FileNotFoundError(f"推理服务器文件不存在: {inference_server_path}")
    
    # 检查并添加模型目录到Python路径
    if actual_solver_path not in sys.path:
        sys.path.insert(0, actual_solver_path)
        print(f"✅ 已将模型目录添加到Python路径: {actual_solver_path}")
    
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
        try:
            pip_cmd = [sys.executable, "-m", "pip", "install", "-r", requirements_path]
            pip_result = subprocess.run(pip_cmd, capture_output=True, text=True)
            if pip_result.returncode != 0:
                print(f"❌ pip安装失败 (exit code {pip_result.returncode})")
                print(f"STDOUT: {pip_result.stdout}")
                print(f"STDERR: {pip_result.stderr}")
            else:
                print("✅ 依赖安装完成")
                if VERBOSE:
                    print(f"安装输出: {pip_result.stdout}")
        except Exception as e:
            print(f"❌ pip安装过程出错: {e}")
    elif requirements_path and not should_install:
        print("requirements.txt 已找到，但环境依赖齐全，默认跳过 pip 安装。设置 FORCE_PIP_INSTALL=1 可强制执行。")
    else:
        print("未找到requirements.txt，跳过依赖安装")
    
    # 运行推理服务器 + 网关流程
    print("运行评估API推理服务器...")
    try:
        result = subprocess.run([
            sys.executable, "inference_server.py"
        ], capture_output=True, text=True, timeout=3600)  # 1小时超时
        
        print(f"推理服务器退出码: {result.returncode}")
        if result.stdout:
            print("标准输出:")
            print(result.stdout)
        if result.stderr:
            print("标准错误:")
            print(result.stderr)
            
        if result.returncode != 0:
            print(f"❌ 推理服务器返回非零状态 {result.returncode}，请检查上方日志")
        else:
            print("✅ 推理服务器运行成功")
    except subprocess.TimeoutExpired:
        print("❌ 推理服务器运行超时（1小时）")
    except Exception as e:
        print(f"❌ 运行推理服务器时出错: {e}")
    
    # 检查提交文件
    submission_parquet = "/kaggle/working/submission.parquet"
    submission_csv = "/kaggle/working/submission.csv"
    current_dir_submission = os.path.join(os.getcwd(), "submission.parquet")
    current_dir_submission_csv = os.path.join(os.getcwd(), "submission.csv")
    
    found_submission = False
    if os.path.exists(submission_parquet):
        print(f"\n✅ 成功！生成: {submission_parquet}")
        found_submission = True
    elif os.path.exists(current_dir_submission):
        print(f"\n✅ 成功！生成: {current_dir_submission}")
        found_submission = True
        
    if os.path.exists(submission_csv):
        print(f"✅ 同步生成: {submission_csv}")
        found_submission = True
    elif os.path.exists(current_dir_submission_csv):
        print(f"✅ 同步生成: {current_dir_submission_csv}")
        found_submission = True
    
    if found_submission:
        print("📊 模型运行完成")
        print("\n📁 从输出面板下载submission.parquet (或 submission.csv)")
    else:
        print("❌ 未创建submission.parquet - 请检查上面的错误")
        # 提供一些调试信息
        print("当前工作目录内容:")
        try:
            for item in os.listdir("."):
                print(f"  - {item}")
        except Exception as e:
            print(f"无法列出目录内容: {e}")
        
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
finally:
    os.chdir(original_dir)

print("\n🏁 处理完成!")
