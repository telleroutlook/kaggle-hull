# Hull Tactical - Market Prediction 项目概览

## 项目简介

本项目是为参与 Kaggle 竞赛 "Hull Tactical - Market Prediction" 而设立。竞赛的目标是利用机器学习模型预测标普500指数（S&P 500）的超额收益，并在管理波动性约束（120%波动率限制）的同时，力争获得优于标普500指数的回报。

**核心挑战**: 这不仅是对预测建模能力的考验，更是对有效市场假说（Efficient Market Hypothesis, EMH）的一次挑战，试图证明在机器学习时代，市场并非完全有效。

## 竞赛时间线

**训练阶段**:
- 开始日期: 2025年9月16日
- 报名截止: 2025年12月8日
- 团队合并截止: 2025年12月8日
- 最终提交截止: 2025年12月15日

**预测阶段**:
- 竞赛结束日期: 2026年6月16日

**奖金**: 总奖金$100,000，第一名$50,000

## 数据集

数据集位于 `input/hull-tactical-market-prediction/` 目录下，包含以下文件：

*   `train.csv`: 历史市场数据，包含数十年的数据（8992行），早期数据存在大量缺失值
*   `test.csv`: 模拟测试集（12行），结构与未公开的测试集一致。公开排行榜的测试集是训练集最后180个日期ID的副本
*   `kaggle_evaluation/`: 包含评估API所需的文件

### 数据特征（共112个特征列）

训练集和测试集包含多种特征，具体分类：

*   **D1-D9**: 虚拟/二元特征 (9个)
*   **E1-E20**: 宏观经济特征 (20个)
*   **I1-I9**: 利率特征 (9个)
*   **M1-M18**: 市场动态/技术指标特征 (18个)
*   **P1-P13**: 价格/估值特征 (13个)
*   **S1-S12**: 情绪特征 (12个)
*   **V1-V13**: 波动率特征 (13个)
*   **MOM1-MOM9**: 动量特征 (9个)

### 目标变量 (仅训练集)

*   `forward_returns`: 购买并持有一天后标普500指数的收益
*   `risk_free_rate`: 联邦基金利率
*   `market_forward_excess_returns`: 相对于预期的前向收益，通过减去滚动五年平均前向收益并使用中位数绝对偏差（MAD）进行winsorizing计算得出

### 测试集特有字段

*   `is_scored`: 该行是否计入评估指标计算
*   `lagged_forward_returns`: 滞后一天的标普500指数收益
*   `lagged_risk_free_rate`: 滞后一天的联邦基金利率
*   `lagged_market_forward_excess_returns`: 滞后一天的超额收益

## 评估与提交

### 评估指标

竞赛使用一种变体夏普比率作为评估指标，该指标会惩罚那些波动性显著高于基础市场或未能跑赢市场收益的策略。

### 提交要求

*   **提交方式**: 必须通过Kaggle Notebook使用评估API提交
*   **运行时间限制**: 
    - 训练阶段: CPU Notebook ≤ 8小时，GPU Notebook ≤ 8小时
    - 预测阶段: 延长至9小时
*   **网络访问**: 禁用
*   **外部数据**: 允许使用公开可用的外部数据和预训练模型
*   **预测目标**: 对每个交易日预测持有标普500指数的最优资金分配比例（0到2之间）
*   **输出格式**: `submission.parquet` 格式

## 评估API结构

Kaggle评估API的代码位于 `input/hull-tactical-market-prediction/kaggle_evaluation/` 目录中：

### 核心组件

*   `default_gateway.py`: 默认网关实现，负责：
    - 解压数据路径
    - 生成数据批次
    - 调用用户模型进行预测
    - 验证预测结果
    - 写入提交文件

*   `default_inference_server.py`: 默认推理服务器，用于启动用户模型

*   `core/` 目录包含API的核心实现：
    - `base_gateway.py`: 网关基类，包含错误处理和文件共享功能
    - `relay.py`: gRPC通信模块
    - `templates.py`: 模板类定义

### 工作流程

1. 网关读取测试数据并分批处理
2. 通过gRPC调用推理服务器进行预测
3. 验证预测结果的完整性和格式
4. 生成最终的提交文件

## 开发指南

### 模型开发要求

*   **核心目标**: 预测标普500指数的超额收益，输出0-2之间的资金分配比例
*   **约束条件**: 波动率不超过120%
*   **数据限制**: 不能"窥视"未来数据
*   **输出格式**: 必须生成符合API要求的 `submission.parquet` 文件

### 技术实现要点

1. **数据预处理**: 
   - 处理缺失值（早期数据存在大量缺失）
   - 特征标准化/归一化
   - 处理时间序列数据的平稳性

2. **模型策略**:
   - 考虑使用时间序列模型（ARIMA, LSTM, Transformer）
   - 集成学习方法（XGBoost, LightGBM, Random Forest）
   - 考虑波动率约束的风险管理

3. **验证策略**:
   - 使用时间序列交叉验证
   - 考虑市场状态（牛市/熊市）的划分
   - 评估模型在波动率约束下的表现

## Kaggle部署指南

### 1. 创建Kaggle部署包

使用提供的脚本创建可上传到Kaggle的压缩包：

```bash
python create_kaggle_archive.py
```

这将在 `input/` 目录下生成 `kaggle_hull_solver.zip` 文件。

### 2. 上传到Kaggle

1. 登录Kaggle并进入Hull Tactical竞赛页面
2. 创建新的数据集，上传 `kaggle_hull_solver.zip`
3. 在Notebook中连接该数据集

### 3. 运行模型

有两种方式运行模型：

**方法1: 简单单元格版本**
- 将 `kaggle_simple_cell.py` 的内容复制粘贴到单个Kaggle notebook单元格中
- 运行单元格即可自动执行模型

**方法2: 标准方式**
- 在Notebook中解压上传的zip文件
- 运行 `working/main.py`

### 4. 输出文件

- 模型将在 `/kaggle/working/submission.parquet` 生成提交文件
- 确保预测值在0-2之间
- 包含 `date_id` 和 `prediction` 两列

## 快速开始

1. **数据探索**: 分析 `train.csv` 的特征分布、缺失值模式和相关性
2. **建模实验**: 构建基础模型进行预测
3. **API集成**: 按照 `kaggle_evaluation/` 中的示例集成评估API
4. **本地测试**: 使用提供的测试集验证模型性能
5. **提交优化**: 调整模型参数和特征工程策略
6. **Kaggle部署**: 使用上述指南打包和提交模型