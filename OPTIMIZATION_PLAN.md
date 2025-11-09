# Score Regression Recovery Plan (2025-11-09)

参考基线提交 a583879d9fff243cb96cbee09892bcc1e9ef976d。

## 1. 诊断结论

- **训练/推理配置漂移**：`train_experiment.py` 强制 `augment_data=True` 且 `standardize=True`（working/train_experiment.py:127-141），而 `inference_server.py` 以默认值初始化 `FeaturePipeline()` 并调用 `load_train_data()`（无增强，working/inference_server.py:103-110）。生成的 OOF artefact 因此记录了与线上模型不同的数据分布。
- **杠杆仍沿用过期 OOF**：推理入口在检测到 artefact 后直接复用 `preferred_scale=40`，且不会重新调参（working/inference_server.py:128-148）。日志显示 Sharpe=0.0000，说明新模型的尺度完全未校准。
- **滚动/技术指标缺乏状态管理**：`FeaturePipeline` 在 `transform` 内大量使用 `rolling()`（工作集: working/lib/features.py:205-360），但线上推理是按批次流式调用，窗口状态每个 batch 重置，导致推理期特征严重失真。

## 2. 优先级路线图

### 2.1 同步训练与推理管线（P0）
- [ ] 提供 `build_feature_pipeline(**kwargs)` 与 `load_training_frame(*, augment: bool)` 工具，让 CLI、OOF、推理共用同一份配置。默认关闭增广，必要时通过环境变量 `HULL_AUGMENT_DATA=1` 触发。
- [ ] 在 OOF artefact 中记录 `pipeline_config_hash`、`augment_flag`。推理端读取 artefact 后若不匹配则回退到本地 `optimize_scale_with_rolling_cv`。
- [ ] 为 `FeaturePipeline` 的 constructor 加入 `from_config()`，避免参数硬编码在多个入口。

### 2.2 杠杆与 overlay 自动再校准（P0）
- [ ] 当 artefact 中的 Sharpe 或校准时间早于最近一次代码 hash 时，强制重新调参并更新 `/kaggle/working/artifacts/oof_summary.json`。
- [ ] 推理端增加 `HULL_FORCE_RECALIBRATE=1` 以便在打包后自动刷新 scale/overlay，并将新 artefact 拷贝回 `working/artifacts/`。

### 2.3 滚动特征在线一致性（P1）
- [ ] 为 `FeaturePipeline` 增加 stateful 模式：在 `transform()` 里维护 `deque` 缓存，使 rolling 指标使用跨 batch 历史。
- [ ] 对技术指标/交互特征添加单元测试，验证逐批输入与一次性输入的输出一致性（允差 < 1e-6）。

### 2.4 Artefact 再生成与验证（P1）
- [ ] 以与推理完全相同的配置重新运行 `train_experiment.py`，写回新的 `working/artifacts/oof_summary.json`。
- [ ] 度量项需至少包含：OOF Sharpe、均值/std、std_guard 触发率、overlay 命中率。写入 `working/hull_metrics.csv` 供回归分析。
- [ ] 在 `create_kaggle_archive.py` 中新增 artefact 过期检测（时间戳+哈希），防止旧 artefact 被打包。

### 2.5 验证
- [ ] 本地运行 `kaggle_simple_cell_fixed.py`，确认日志中 `🎯 Calibrated allocation scale` 不再是 0.0000，并且 scale 来源与当前代码一致。
- [ ] 最少 2 次提交（OOF 更新前/后）验证 public LB 的差异，将结果写入 `hull_metrics.csv`。

## 3. 预期成果

- 训练/推理配置统一，可复现 OOF 指标。
- 杠杆自动回落到最新模型（无人工更新 artefact 也不会降分）。
- 滚动特征在流式推理中与离线表现一致，预测标准差恢复到 ≥0.001。
- 新的 artefact 版本化 & 自检，避免再次引用过期参数。
