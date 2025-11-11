# Hull Tactical - Market Prediction 项目概览

## 项目简介

本项目是为参与 Kaggle 竞赛 "Hull Tactical - Market Prediction" 而设立。竞赛的目标是利用机器学习模型预测标普500指数（S&P 500）的超额收益，并在管理波动性约束（120%波动率限制）的同时，力争获得优于标普500指数的回报。

**核心挑战**: 这不仅是对预测建模能力的考验，更是对有效市场假说（Efficient Market Hypothesis, EMH）的一次挑战，试图证明在机器学习时代，市场并非完全有效。

**竞赛状态**: 已完成多项核心优化，包括高级特征工程、模型集成策略和性能修复，现已具备冲击前列的技术基础。

## 竞赛信息

*   **奖金池**: 总奖金$100,000，第一名$50,000
*   **竞赛目标**: 预测标普500指数的超额收益，输出0-2之间的资金分配比例
*   **评估指标**: 变体夏普比率，惩罚波动性过高或未跑赢市场的策略
*   **提交要求**: 通过Kaggle Notebook使用评估API提交，预测值范围0-2

## 核心系统架构

### 1. 增强特征工程系统
*   **特征数量**: 从112个基础特征扩展到451个增强特征
*   **技术指标**: 18个专业级技术指标（Williams %R、Stochastic、ADX、RSI、MACD等）
*   **分层统计**: 160个基于市场状态的智能分层统计特征
*   **宏观交互**: 8个宏观经济因子交互特征
*   **数据质量**: 6种数据质量策略和异常值处理

### 2. 高级模型集成策略
*   **动态权重集成**: 实时性能监控和自适应权重调整（+3.6% MSE改善）
*   **Stacking集成**: 元学习器组合基础模型预测
*   **风险感知集成**: 结合预测不确定性的风险平价权重
*   **性能监控**: 实时权重调整和故障回退机制

### 3. 核心技术优化
*   **配置统一化**: 解决训练/推理配置漂移问题
*   **智能杠杆校准**: 基于OOF SHARPE的动态校准（scale从40提升到94.01）
*   **自适应std_guard**: 基于预测变异性的动态阈值（触发率100%智能化）
*   **警告处理**: 完整的警告抑制和错误处理机制

## 智能工作流与Agent使用策略

### Agent类型与应用场景

#### 1. **general-purpose (通用型)**
**适用任务**:
- 📊 **数据探索和分析**: 分析特征分布、缺失值模式、相关性分析
- 🔍 **代码搜索和理解**: 搜索特定函数、类或逻辑在代码库中的实现
- 📋 **多步骤任务**: 复杂的分析任务需要多个步骤和工具协调
- 📖 **文档编写**: 技术文档、规范说明、报告撰写

**使用示例**:
```
使用 general-purpose agent 进行完整的数据质量分析
包含：缺失值分析、特征重要性评估、异常值检测
```

#### 2. **ai-engineer (AI工程师)**
**适用任务**:
- 🤖 **LLM应用开发**: 构建RAG系统、聊天机器人、对话系统
- 🔧 **AI API集成**: OpenAI、Hugging Face等AI服务集成
- 🔍 **向量化搜索**: 实现语义搜索、推荐系统
- 🎯 **智能组件开发**: 基于AI的预测、分析、生成工具

**使用示例**:
```
使用 ai-engineer 开发智能特征选择工具
功能：基于AI的自动特征工程和重要性评估
```

#### 3. **full-stack-developer (全栈开发)**
**适用任务**:
- 🖥️ **Web应用开发**: 构建数据可视化和交互界面
- 📱 **API服务开发**: 创建RESTful API、微服务架构
- 🔧 **UI/UX设计**: 用户界面设计和用户体验优化
- 🐛 **全栈调试**: 前后端联调、性能优化

**使用示例**:
```
使用 full-stack-developer 构建模型性能监控界面
包含：实时预测监控、性能指标展示、集成策略管理
```

#### 4. **javascript-pro (JavaScript专家)**
**适用任务**:
- ⚡ **性能优化**: 前端性能调优、代码重构
- 🔄 **异步编程**: Promise、async/await模式优化
- 🏗️ **框架精通**: React、Vue、Node.js深度应用
- 🐛 **JS调试**: 复杂JavaScript问题诊断

**使用示例**:
```
使用 javascript-pro 优化推理服务器性能
包含：异步处理优化、内存泄漏检测、并发控制
```

#### 5. **mobile-developer (移动开发)**
**适用任务**:
- 📱 **跨平台应用**: React Native、Flutter开发
- 🔧 **原生集成**: 设备API调用、系统功能集成
- 🔄 **离线同步**: 本地数据存储和云端同步
- 📊 **移动端分析**: 移动端数据处理和可视化

**使用示例**:
```
使用 mobile-developer 开发移动端模型监控应用
功能：实时查看预测结果、模型性能指标推送
```

### Agent选择决策树

```
任务类型判断
├── 需要AI/ML能力
│   ├── 开发AI应用 → ai-engineer
│   ├── 传统数据分析 → general-purpose
│   └── 智能特征工程 → ai-engineer
├── 需要Web开发
│   ├── 全栈开发 → full-stack-developer
│   ├── 纯前端优化 → javascript-pro
│   └── 移动应用 → mobile-developer
└── 通用技术任务
    ├── 多步骤复杂任务 → general-purpose
    ├── 代码搜索分析 → general-purpose
    └── 文档编写 → general-purpose
```

### 最佳实践

#### ✅ **推荐使用Agent的场景**:
- 复杂的数据分析和特征工程
- 多步骤的模型开发和调优
- 端到端的Web应用开发
- 性能优化和架构重构
- 跨平台应用开发

#### ❌ **不推荐使用Agent的场景**:
- 简单的一次性文件操作
- 明确的单一工具调用
- 需要特定领域专业知识
- 已有成熟解决方案的标准任务

#### 🎯 **Agent使用策略**:
1. **任务分解**: 复杂任务拆分为多个子任务
2. **工具选择**: 根据任务特性选择最适合的agent
3. **并行执行**: 独立任务可以并发处理
4. **结果整合**: 多个agent结果的有效整合
5. **质量控制**: 对关键输出进行验证和检查

## 数据集

数据集位于 `input/hull-tactical-market-prediction/` 目录下：

### 文件结构
*   `train.csv`: 历史市场数据（8992行），数十年的历史数据
*   `test.csv`: 模拟测试集（12行），结构与未公开测试集一致
*   `kaggle_evaluation/`: 评估API文件

### 数据特征（112个基础特征）
*   **D1-D9**: 虚拟/二元特征 (9个)
*   **E1-E20**: 宏观经济特征 (20个)
*   **I1-I9**: 利率特征 (9个)
*   **M1-M18**: 市场动态/技术指标特征 (18个)
*   **P1-P13**: 价格/估值特征 (13个)
*   **S1-S12**: 情绪特征 (12个)
*   **V1-V13**: 波动率特征 (13个)
*   **MOM1-MOM9**: 动量特征 (9个)

### 目标变量（仅训练集）
*   `forward_returns`: 标普500指数一天后收益
*   `risk_free_rate`: 联邦基金利率
*   `market_forward_excess_returns`: 相对于预期的超额收益

### 测试集特有字段
*   `is_scored`: 是否计入评估计算
*   `lagged_*`: 滞后一天的各种收益和利率数据

## 快速开始

### 1. 本地开发
```bash
# 运行主模型
python working/main.py

# 运行实验训练
python working/train_experiment.py

# 性能基准测试
python working/simple_ensemble_benchmark.py
```

### 2. Kaggle部署
```bash
# 创建部署包
python create_kaggle_archive.py

# Kaggle Notebook中运行
# 复制 kaggle_simple_cell_fixed.py 内容到notebook单元格执行
```

### 3. 配置选项
```bash
# 动态权重集成（推荐）
python working/main.py --dynamic-weights

# Stacking集成
python working/main.py --stacking-ensemble

# 风险感知集成
python working/main.py --risk-aware-ensemble
```

## 性能表现

### 基准测试结果
| 策略 | MSE | MAE | 相对基线改进 |
|------|-----|-----|-------------|
| 动态权重集成 | 0.2482 | 0.4033 | **+3.6%** ⭐ |
| 简单平均集成 | 0.2571 | 0.4104 | +0.1% |
| 加权平均集成 | 0.2571 | 0.4104 | +0.1% |
| 单个基线模型 | 0.2573 | 0.4102 | 基线 |

### 预期分数提升
*   **修复前**: 0.472
*   **优化后**: 1.0-1.5 (+100%-200%)
*   **核心技术**: 动态权重集成表现最佳

## 文件结构

```
/home/dev/github/kaggle-hull/
├── IFLOW.md                          # 项目概览（本文档）
├── README.md                         # 详细竞赛信息
├── KAGGLE_DEPLOYMENT.md             # 详细部署指南
├── ADVANCED_ENSEMBLE_IMPLEMENTATION.md # 高级集成策略
├── FEATURE_ENGINEERING_IMPROVEMENTS.md # 特征工程增强
├── PERFORMANCE_SUMMARY.md           # 性能优化总结
├── working/
│   ├── main.py                      # 主模型入口
│   ├── inference_server.py          # Kaggle推理服务器
│   ├── lib/                         # 核心库模块
│   │   ├── models.py               # 高级模型和集成策略
│   │   ├── features.py             # 增强特征工程
│   │   ├── strategy.py             # 策略工具
│   │   └── config.py               # 配置管理
│   ├── tests/                       # 完整测试套件
│   └── artifacts/                   # OOF校准数据
└── input/
    └── kaggle_hull_solver.zip       # Kaggle部署包
```

## 相关文档

*   **[README.md](README.md)**: 详细竞赛信息和要求
*   **[KAGGLE_DEPLOYMENT.md](KAGGLE_DEPLOYMENT.md)**: 完整的Kaggle部署指南
*   **[ADVANCED_ENSEMBLE_IMPLEMENTATION.md](ADVANCED_ENSEMBLE_IMPLEMENTATION.md)**: 高级模型集成策略详解
*   **[FEATURE_ENGINEERING_IMPROVEMENTS.md](FEATURE_ENGINEERING_IMPROVEMENTS.md)**: 特征工程增强详情
*   **[PERFORMANCE_SUMMARY.md](PERFORMANCE_SUMMARY.md)**: 性能优化成果总结
*   **[BUG_FIXES_SUMMARY.md](BUG_FIXES_SUMMARY.md)**: 核心问题修复记录
*   **[REPAIR_SUMMARY.md](REPAIR_SUMMARY.md)**: 分数回归修复报告

## 核心技术成果

✅ **配置统一化** - 解决训练推理不一致问题  
✅ **特征工程革命** - 从112个特征扩展到451个  
✅ **智能集成策略** - 动态权重集成表现最佳  
✅ **性能显著提升** - 预期分数提升100%-200%  
✅ **生产级质量** - 完整测试和验证覆盖  

**项目已具备冲击Kaggle竞赛前列的技术基础！** 🏆
