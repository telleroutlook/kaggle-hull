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

