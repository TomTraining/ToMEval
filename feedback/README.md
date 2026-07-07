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
feedback_output/                   # config.yaml 的 output_path（默认 feedback_output）
└── _intermediate/
    ├── bad_cases/            # Stage 1 加载的 bad case（按维度分层抽样）
    ├── diagnosis_reports/    # Stage 2 诊断报告 + dimension_coverage.json
    ├── synth_raw/            # Stage 3 原始合成候选（未过 LSH）
    └── synth_clean/          # Stage 4 过 LSH 守门员后的最终训练数据
```

---

## 流水线阶段

阶段由 `config.yaml` 的 `stage` 字段控制（不是 CLI 参数）：

| Stage | `stage` 取值 | 说明 |
|---|---|---|
| 1 | `load` | 从 `predictions_root`（默认 `results`）加载各模型并集 bad case，按维度分层 |
| 2 | `diagnose` | 按能力维度对 bad case 分组，合成模型逐批生成 `DimensionDiagnosisReport` |
| 3 | `synth` | 读诊断报告生成新样本（含 LSH 去泄） |
| 4 | `dedupe` | LSH 守门员过滤测试集泄漏（`synth` 已包含此步） |
| 全流程 | `all` | 上述顺序执行 |

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
| `max_bad_cases` | `0`（不限） | 每数据集最多 bad case 数 |

（数据集通过 `synthesis_datasets` 列表逐项指定，见下方配置说明。）

---

## 配置文件说明（config.yaml）

```yaml
# 合成模型（诊断 / 合成用，建议强模型）
synthesis_model:
  api_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  api_key: <DashScope API key>
  model_name: deepseek-v4-flash
  temperature: 0.6
  max_workers: 16

# 评估结果根目录（Stage 1 从这里读预测）
predictions_root: results

# 守门员（LSH）
leakage_guard:
  test_root: ./datasets           # 相对于 ToMEval/ 目录
  threshold: 0.6                  # 与测试集的 Jaccard 相似度阈值
  internal_threshold: 0.85        # 合成样本之间的内部去重阈值
  num_perm: 128
  ngram: 4

# 合成参数
synthesis:
  max_retries_per_diagnosis: 3    # 生成失败时最多重试次数
  bad_cases_per_report: 1         # 每个诊断报告输入几条 bad case
  diagnosis_batch_size: 10        # stage2 每批诊断报告数
  synthesis_batch_size: 20        # stage3 每批合成数据数

# 数据集列表（每数据集级指定 target_samples / samples_per_report / models）
synthesis_datasets:
  - name: ToMBench
    target_samples: 50
    samples_per_report: 5
    models:
      - name: <model_name>
        exp: ~                    # ~ 表示自动取最新 exp_*
```

---

## 支持的数据集

ToMBench / SocialIQA / BigToM / EmoBench / FanToM / HiToM / **SocialMind**。
合成 schema 注册表在 `stage3_synthesis.py`，维度字段映射在 `stage1_load_predictions.py:get_dimension_key`。

### SocialMind（中文，4 题型异构）

SocialMind 一个样本含 4 种题型:Q1 单选(1 正 3 误)/ Q2 多选(≥2 正)/ Q3 判断(是·否·无法确定)/ Q4 开放(rubric 评分)。
单条 dataset 级 format 装不下,所以**按 qtype 路由**:

- 维度键编成 `dim__qtype__zh`(如 `1.1.1__Q1__zh`),每份诊断报告题型同质;
- `prompts.py:SOCIALMIND_FORMAT_BY_QTYPE` 与 `stage3_synthesis.py:SOCIALMIND_SCHEMA_BY_QTYPE` 各按 qtype 给 4 套 format/schema;
- qtype 用 `report["_meta"]["dimension"]`(stage2 程序化写入的权威键)解析,**不读 LLM 回显的顶层 `dimension`**;
- Q4 强制 `meta_dim` = 诊断维度,以命中 `tasks/SocialMind/q4_judge_prompts.json` 的 rubric 键。

> ⚠️ **维度×题型桶爆炸(用 SocialMind 必读)**
>
> 合成预算是 `报告数 = target_samples / samples_per_report`,这些报告按"桶"(分组键)分配,**桶里有错题才会被合成**。SocialMind 的桶键是 `dim__qtype__zh`,所以**桶数 = 维度 × 题型**:71 个三级维度 × 4 题型 ≈ **最多 284 个桶**(其它数据集只按维度分组,桶数少一个量级)。
>
> 当 **桶数 > 报告数** 时,`stage2_diagnosis.py:allocate_reports_by_dimension`(L196-201)只给错误率最高的 **top-N 个桶各 1 份**,其余桶 **0 份、一条都合成不出来**。例:`target_samples=200, samples_per_report=5` → 仅 40 份报告 → 覆盖 ≤40 个桶,240+ 个 (维度,题型) 组合全空,Q4(错题本就少)最易被饿死,覆盖严重不均。
>
> **缓解**:
> 1. 把 `target_samples` 调大(几百~上千),让报告数 ≥ 桶数,每桶至少摊到 1 份;
> 2. 或分题型多轮跑,每轮配置只让一种 qtype 参与,把预算集中;
> 3. 若只关心高频出错维度,接受 top-N 自动聚焦即可。

## 添加新数据集

1. 在 `config.yaml` 的 `synthesis_datasets` 中添加 `- name: MyDataset`。
2. 在 `stage3_synthesis.py` 的 `SYNTHESIS_SCHEMA_REGISTRY` 中注册对应 Pydantic schema(题型异构的数据集参考 SocialMind 的 qtype 路由写法)。
3. 在 `prompts.py` 的 `SYNTHESIS_FORMAT_REGISTRY` + `DATASET_SKILL_REGISTRY` 中补 format/技能说明。
4. 在 `stage1_load_predictions.py:get_dimension_key()` 中添加维度字段映射。
5. 在 `stage4_lsh_filter.py:_KNOWN_TASKS` 中加入数据集名(否则其测试集不被索引、防泄漏失效)。

---

## 常见问题

**diagnose 阶段找不到 bad cases？**
先跑 `--stage load`，再跑 `--stage diagnose`。

**synth 阶段找不到诊断报告？**
检查 `data_output/diagnosis_reports/<dataset>/<split>/dimension_reports.jsonl` 是否存在。
