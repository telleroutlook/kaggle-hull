# Hull Tactical - Market Prediction Kaggle 部署指南

## 概述

本文档指导您如何将Hull Tactical - Market Prediction项目打包并部署到Kaggle平台进行竞赛提交。

## 文件结构

项目包含以下关键文件用于Kaggle部署：

```
├── create_kaggle_archive.py    # 创建Kaggle部署包的脚本
├── kaggle_simple_cell.py       # 单元格Kaggle notebook版本
├── kaggle_optimized_cell.py    # 优化版Kaggle notebook版本
├── working/
│   ├── main.py                 # 主模型入口点
│   ├── config.ini              # 配置文件
│   ├── __init__.py             # 包初始化文件
│   ├── lib/                    # 模块化库
│   │   ├── __init__.py
│   │   ├── env.py              # 环境检测
│   │   ├── data.py             # 数据加载
│   │   ├── features.py         # 特征工程
│   │   ├── models.py           # 模型定义
│   │   ├── evaluation.py       # 评估指标
│   │   ├── utils.py            # 工具函数
│   │   └── config.py           # 配置管理
│   └── tests/                  # 测试套件
├── requirements.txt            # Python依赖
├── README.md                   # 项目说明
├── IFLOW.md                    # 项目概览和开发指南
└── input/
    └── kaggle_hull_solver.zip  # 生成的部署包
```

## 部署步骤

### 1. 创建部署包

运行打包脚本生成Kaggle部署包：

```bash
python3 create_kaggle_archive.py
```

这将在 `input/` 目录下生成 `kaggle_hull_solver.zip` 文件。

### 2. 上传到Kaggle

1. 登录 [Kaggle](https://www.kaggle.com)
2. 进入 [Hull Tactical - Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction) 竞赛页面
3. 点击右上角的 "+ Add Data" 按钮
4. 选择 "Upload" 选项
5. 上传生成的 `kaggle_hull_solver.zip` 文件
6. 为数据集命名（例如 "hull-tactical-solver"）
7. 点击 "Create" 创建数据集

### 3. 创建Kaggle Notebook

1. 在竞赛页面点击 "Notebooks" 标签
2. 点击 "New Notebook" 按钮
3. 在 "Data" 面板中，找到并添加你刚刚上传的数据集
4. 确保数据集已连接（显示绿色勾号）

### 4. 运行模型

#### 方法1：优化版单元格（推荐）

使用 `kaggle_optimized_cell.py` 中的代码，在Notebook中创建一个新的代码单元格：

```python
# Kaggle环境模型部署优化版
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
```

#### 方法2：简单单元格版本

1. 在Notebook中创建一个新的代码单元格
2. 将 `kaggle_simple_cell.py` 文件的全部内容复制粘贴到该单元格
3. 运行单元格

#### 方法3：命令行方式

1. 解压数据集：
   ```python
   !unzip -q /kaggle/input/your-dataset-name/kaggle_hull_solver.zip -d /kaggle/working/
   ```

2. 运行模型：
   ```python
   !cd /kaggle/working && python working/main.py
   ```

### 5. 提交结果

1. 运行完成后，检查是否生成了 `/kaggle/working/submission.csv` 文件
2. 点击Notebook右上角的 "Save Version" 按钮
3. 选择 "Save & Run All (Commit)"
4. 等待运行完成，然后点击 "Submit" 按钮提交结果

## 配置文件

项目包含 `working/config.ini` 配置文件，允许你调整模型参数：

```ini
[model]
type = "baseline"
baseline_n_estimators = 100
baseline_max_depth = 10
baseline_random_state = 42

[features]
max_features = 20
rolling_windows = [5, 10, 20]
lag_periods = [1, 2, 3]

[evaluation]
volatility_constraint = 1.2
risk_free_rate = 0.0

[logging]
level = "INFO"
format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## 模型开发

### 修改主模型

编辑 `working/main.py` 文件来实现你的预测逻辑。关键部分包括：

```python
# 加载数据
test_data = pd.read_csv(test_csv_path)

# 实现预测逻辑
# 这里应该是你的模型代码
predictions = your_model.predict(test_data)

# 确保预测值在0-2之间
predictions = np.clip(predictions, 0, 2)

# 创建提交文件
submission_df = pd.DataFrame({
    'date_id': test_data['date_id'],
    'prediction': predictions
})
```

### 添加依赖

在 `requirements.txt` 中添加你的模型所需的Python包：

```txt
your-package-name>=1.0.0
another-package>=2.0.0
```

### 添加额外文件

如果需要添加额外的Python模块：
- 将文件放在 `working/` 目录下
- 确保在 `create_kaggle_archive.py` 的 `files_to_include` 列表中添加相应路径

## 调试技巧

### 常见问题

1. **找不到模型文件**
   - 确保数据集已正确连接
   - 检查数据集名称是否与脚本中搜索的路径匹配

2. **依赖安装失败**
   - 在 `kaggle_optimized_cell.py` 中添加需要的pip安装命令
   - 确保版本兼容性

3. **内存不足**
   - 优化模型内存使用
   - 考虑使用更轻量的算法
   - 使用数据分块处理

4. **运行超时**
   - 训练阶段：8小时限制
   - 预测阶段：9小时限制
   - 优化代码性能

### 本地测试

在部署到Kaggle之前，建议先在本地测试：

```bash
# 在项目根目录运行
python working/main.py --verbose
```

## 性能优化建议

1. **内存优化**
   - 特征工程时限制特征数量（通过配置文件）
   - 使用数据分块处理
   - 及时释放不需要的变量
   - 使用适当的数据类型

2. **速度优化**
   - 向量化操作替代循环
   - 使用更高效的算法
   - 启用GPU加速（如果可用）

3. **模型优化**
   - 特征选择减少维度
   - 使用轻量级模型
   - 模型压缩技术

## 配置管理

通过 `working/lib/config.py` 模块管理配置：

```python
from lib.config import get_config

config = get_config()
model_config = config.get_model_config()
features_config = config.get_features_config()
```

## 测试套件

项目包含完整的测试套件，位于 `working/tests/` 目录下：

- `test_env.py` - 环境检测测试
- `test_data.py` - 数据加载测试
- `test_features.py` - 特征工程测试
- `test_models.py` - 模型测试
- `test_utils.py` - 工具函数测试
- `test_evaluation.py` - 评估指标测试
- `simple_test.py` - 简化测试

运行测试：
```bash
python working/tests/simple_test.py
```

## 注意事项

- 确保模型不在训练阶段"窥视"未来数据
- 预测值必须在0-2之间
- 提交文件必须是 `submission.csv` 格式
- 包含 `date_id` 和 `prediction` 两列

## 故障排除

如果遇到问题：

1. 检查Kaggle notebook的输出日志
2. 验证数据集路径是否正确
3. 确保所有依赖都已安装
4. 检查模型输出格式
5. 查看竞赛论坛获取帮助

祝你好运！🚀