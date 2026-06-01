# 数据合成模块

本模块实现 Theory-of-Mind 训练数据的合成流水线。入口脚本位于项目根目录 `run_feedback.py`。

---

## 文件结构

```
feedback/
├── config.yaml                 # 配置（合成模型 / 数据集 / 各阶段参数）
├── stage1_load_predictions.py  # Stage 1：从已有预测文件加载 bad case（按维度分层抽样）
├── stage2_diagnosis.py         # Stage 2：维度批量诊断
├── stage3_synthesis.py         # Stage 3：从诊断报告合成新样本
├── stage4_lsh_filter.py        # Stage 4：LSH 守门员去测试集泄漏
└── prompts.py                  # 诊断 / 合成 Prompt 模板

run_feedback.py          # 入口脚本（项目根目录）
```

> **训练集口径**：进入训练集的样本必须满足 LSH 守门员通过（无测试集泄漏）。`synth_clean/` 下的 parquet 是最终可用数据，`synth_raw/` 是中间产物。难度过滤由 `filter/` 统一处理。

```
data_output/
├── bad_cases/            # Stage 1 加载的 bad case（按维度分层抽样）
├── diagnosis_reports/    # Stage 2 诊断报告 + dimension_coverage.json
├── synth_raw/            # Stage 3 原始合成候选（未过 LSH）
└── synth_clean/          # Stage 4 过 LSH 守门员后的最终训练数据
```

---

## 流水线阶段

| Stage | 入口参数 | 说明 |
|---|---|---|
| 1 | `--stage load` | 从 `tomeval_predictions_latest_full` 加载三模型并集 bad case，按维度分层 |
| 2 | `--stage diagnose` | 按能力维度对 bad case 分组，合成模型逐批生成 `DimensionDiagnosisReport` |
| 3 | `--stage synth` | 读诊断报告生成新样本（含 LSH 去泄） |
| 4 | `--stage dedupe` | LSH 守门员过滤测试集泄漏（`synth` 已包含此步） |
| 全流程 | `--stage all` | 上述顺序执行 |

---

## 快速开始

所有运行参数在 `feedback/config.yaml` 中配置，直接运行：

```bash
python run_feedback.py
# 或指定配置文件
python run_feedback.py --config feedback/config.yaml
```

config.yaml 中的运行控制字段：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `stage` | `all` | `all` / `load` / `diagnose` / `synth` / `dedupe` |
| `dataset` | `""`（全部） | 只运行单个数据集，如 `ToMBench` |
| `max_bad_cases` | `0`（不限） | 每数据集最多 bad case 数 |
| `iteration` | `1` | 迭代轮次，影响输出文件命名 |

---

## 配置文件说明（config.yaml）

```yaml
# 合成模型（诊断 / 合成用，建议强模型）
synthesis_model:
  api_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  api_key: <DashScope API key>
  model_name: deepseek-v4-flash
  temperature: 0.8
  max_workers: 16

# 守门员（LSH）
leakage_guard:
  test_root: ./dataset
  threshold: 0.6
  num_perm: 128
  ngram: 4

# 合成参数
synthesis:
  samples_per_report: 3           # 每份维度报告生成的新样本数
  max_retries_per_diagnosis: 3
```

---

## 支持的数据集

ToMBench / SocialIQA / BigToM / EmoBench / FanToM / HiToM。
合成 schema 注册表在 `stage3_synthesis.py`，维度字段映射在 `stage2_diagnosis.py:get_dimension_key`。

## 添加新数据集

1. 在 `config.yaml` 的 `synthesis_datasets` 中添加 `- name: MyDataset`。
2. 在 `stage3_synthesis.py` 的 schema registry 中注册对应 Pydantic schema。
3. 在 `stage2_diagnosis.py:get_dimension_key()` 中添加维度字段映射。

---

## 常见问题

**diagnose 阶段找不到 bad cases？**
先跑 `--stage load`，再跑 `--stage diagnose`。

**synth 阶段找不到诊断报告？**
检查 `data_output/diagnosis_reports/<dataset>/<split>/dimension_reports.jsonl` 是否存在。
