# 优化方案追踪

> 约定：当某项工作完成时，请将该项的“状态”更新为“已完成”，便于全队同步。

## 1. Bug 修复

- [x] **FeaturePipeline 中 fill_value 未定义导致小样本推理崩溃**  
  - 位置：`working/lib/features.py:160-166`  
  - 行动：在未使用分位裁剪的分支引用已有的填充值或新增局部变量，并补充最小样本单测。  
  - 状态：已完成

- [x] **main_fixed.py 输出随机预测易被误用为正式入口**  
  - 行动：从打包清单移除该文件，或在 README/Kaggle 指南中明确注明“示例脚本”并阻止默认执行。  
  - 状态：已完成

- [x] **CLI 与推理服务使用不同默认模型类型**  
  - 行动：统一 `resolve_model_type` 的默认值，或在日志中明确打印并强制同步 `HULL_MODEL_TYPE`。  
  - 状态：已完成

## 2. 数据与特征工程

- [x] **CSV 读取效率与内存占用偏高**  
  - 行动：为 `load_train_data`/`load_test_data` 指定 `dtype`/`usecols`，必要时切换到 `polars` 或分块读取，并减少冗长日志。  
  - 状态：已完成

- [x] **FeaturePipeline 与 engineer_features 双轨实现**  
  - 行动：让离线/在线流程共享同一管线，将 `get_feature_columns` 提取为单一实现，消除漂移。  
  - 状态：已完成

- [x] **滚动统计性能优化**  
  - 行动：考虑 `numba`/`bottleneck`/`polars` 实现，并缓存 `FeaturePipeline` 输出列以减少重复 fit。  
  - 状态：已完成

## 3. 建模与策略

- [x] **主流程缺少 OOF 校验**  
  - 行动：将 `train_experiment.py` 的 TimeSeriesSplit 结果固化为 artefact，并在提交前读取以复用杠杆/指标。  
  - 状态：已完成

- [x] **集成与确定性训练**  
  - 行动：扩展 `HullModel` 支持多模型加权、统一 random_state/early stopping，保证线上线下一致。  
  - 状态：已完成

- [x] **VolatilityOverlay 监控维度不足**  
  - 行动：记录 overlay 缩放因子和触发频率至 `hull_metrics.csv`，探索自适应目标波动率。  
  - 状态：已完成

## 4. 工程与部署

- [x] **Kaggle 单元格脚本输出与内存风险**  
  - 行动：去除 `capture_output=True`，改为流式打印；提供 `VERBOSE`/`FORCE_PIP_INSTALL` 开关，减少 notebook 噪音。  
  - 状态：已完成

- [x] **打包与文档一致性**  
  - 行动：让 `create_kaggle_archive.py` 支持按需包含 tests 并生成 checksum，同时在 README/KAGGLE_DEPLOYMENT 标注 `HULL_MODEL_TYPE` 用法。  
  - 状态：已完成

- [x] **测试矩阵覆盖不足**  
  - 行动：新增针对 `FeaturePipeline` 边界、`VolatilityOverlay` lag 模式、`inference_server.predict` 空批次的 pytest，并连接 CI。  
  - 状态：已完成

## 5. 分数回归排查（2025-11）

- [ ] **预测分布塌缩导致Sharpe为零**  
  - 证据：`working/hull_metrics.csv` 最新几次运行的 `std_prediction` 从基线的 ~0.60 降到 0.06，`working/submission.csv` 也只有 0.91~1.12 的极窄区间。  
  - 行动：在 `train_experiment.py` 中记录/绘制 raw preds 与最终 allocation 的标准差，并在主流程加入阈值守护（例如 `std_prediction < 0.15` 时强制回退到更高的 leverage/换模型）以恢复有效波动。  
  - 状态：待处理

- [ ] **训练缺失 lag 特征导致线上线下特征空间不一致**  
  - 证据：`input/hull-tactical-market-prediction/test.csv` 含有 `lagged_forward_returns` 等列，但 `train.csv` 没有，现有 `FeaturePipeline` 在 fit 阶段无法学习到这些强信号。  
  - 行动：在 `load_train_data` 或 `FeaturePipeline` 中显式构造同名 lag 列（例如用 `forward_returns.shift(1)`），并在单测中验证离线/在线列集合一致。  
  - 状态：待处理

- [ ] **OOF 杠杆与 overlay 参数需与提交版本同步**  
  - 证据：当前 `main.py`/`inference_server.py` 每次重新拟合并用全量数据调 `allocation_scale`，但 Kaggle 打分窗口只有 10 天，导致 scale 与 overlay 完全过拟合历史。  
  - 行动：利用 `train_experiment.py` 在滚动窗口上导出 `preferred_scale`、overlay 配置和 holdout Sharpe，并在提交前强制读取该 artefact 而不是实时重算；追加回测脚本比较“OOF vs. Kaggle 测试窗口”的收益波动差异。  
  - 状态：待处理
