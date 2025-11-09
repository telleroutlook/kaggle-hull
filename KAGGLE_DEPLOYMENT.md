# Hull Tactical - Market Prediction Kaggle 部署指南

## 概述

本文档指导您如何将Hull Tactical - Market Prediction项目打包并部署到Kaggle平台进行竞赛提交。

## 文件结构

项目包含以下关键文件用于Kaggle部署：

```
├── create_kaggle_archive.py    # 创建Kaggle部署包的脚本
├── kaggle_simple_cell_fixed.py # Kaggle notebook单元格脚本（推荐）
├── working/
│   ├── main.py                 # 主模型入口点
│   ├── inference_server.py     # 评估API推理服务器入口
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
python3 create_kaggle_archive.py [--include-tests]
```

这将在 `input/` 目录下生成 `kaggle_hull_solver.zip` 文件及同名 `.sha256` 校验文件。默认为减小体积会跳过 `working/tests` 目录，如需包含测试可添加 `--include-tests`。

### 2. 上传到Kaggle

1. 登录 [Kaggle](https://www.kaggle.com)
2. 进入 [Hull Tactical - Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction) 竞赛页面
3. 点击右上角的 "+ Add Data" 按钮
4. 选择 "Upload" 选项
5. 上传生成的 `kaggle_hull_solver.zip` 文件
6. 为数据集命名（目前推荐 `hull01`）
7. 点击 "Create" 创建数据集

### 3. 创建Kaggle Notebook

1. 在竞赛页面点击 "Notebooks" 标签
2. 点击 "New Notebook" 按钮
3. 在 "Data" 面板中，找到并添加你刚刚上传的数据集（名称若为 `hull01` 即可）
4. 确保数据集已连接（显示绿色勾号）

### 4. 运行模型

#### Kaggle Notebook 单元格（推荐）

1. 在Notebook中创建一个新的代码单元格。
2. 打开仓库根目录的 `kaggle_simple_cell_fixed.py`，复制全部内容。
3. 粘贴到Notebook单元格后运行。脚本会自动：
   - 在 `/kaggle/input` 下递归定位包含 `working/inference_server.py` 的数据集目录；
   - 若检测到 `requirements.txt` 则执行 `pip install -r requirements.txt`；
   - 调用 `working/inference_server.py`，启动官方 `kaggle_evaluation` 网关与推理服务器，生成 `submission.parquet` 及辅助的 `submission.csv`；
   - 将关键日志输出到cell，方便排查。

脚本支持以下环境变量：

- `VERBOSE=1`：打印详细目录/依赖检查信息。
- `FORCE_PIP_INSTALL=1`：即使依赖已满足也强制执行 `pip install`。

> 如需安装额外依赖，可直接在 `kaggle_simple_cell_fixed.py` 中“检查依赖”位置添加 `subprocess.run([sys.executable, '-m', 'pip', 'install', 'pkg'])`。

#### 命令行方式（可选）

1. 解压数据集：
   ```python
   !unzip -q /kaggle/input/your-dataset-name/kaggle_hull_solver.zip -d /kaggle/working/
   ```

2. 运行推理服务器：
   ```python
   !cd /kaggle/working && python working/inference_server.py
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
- `create_kaggle_archive.py` 已默认包含 `working/lib/`、`working/artifacts/` 以及核心入口脚本。若添加新的顶级目录，请同步更新 `build_manifest()`。

OOF artefact（`working/artifacts/oof_summary.json`）会在运行 `python working/train_experiment.py ...` 后生成，并在打包时自动带上。

## HULL_MODEL_TYPE 与 OOF Artefact

- `HULL_MODEL_TYPE` 环境变量用于指定默认模型类型，CLI 与推理服务都会读取该值；也可通过 `--model-type` 显式覆盖。
- 在提交前运行 `working/train_experiment.py` 可生成最新的 TimeSeriesSplit/OOF 指标，`main.py` 与 `inference_server.py` 会在 `--reuse-oof-scale`（默认开启）状态下自动复用 artefact 中记录的杠杆尺度与 Sharpe，以缩短 notebook 调参时间并保持线上/本地一致。
- `working/main_fixed.py` 仅为 notebook 调试脚本，默认拒绝运行随机预测。若确需演示可添加 `--allow-random-baseline`，正式提交务必使用 `main.py`/`inference_server.py`。

## 调试技巧

### 常见问题

1. **找不到模型文件**
   - 确保数据集已正确连接
   - 检查数据集名称是否与脚本中搜索的路径匹配

2. **依赖安装失败**
   - 在 `kaggle_simple_cell_fixed.py` 中添加需要的pip安装命令
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
