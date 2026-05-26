# 数据合成方案（synthesis_plan.md）

> 当前 ToMEval 数据合成的完整技术方案。与 `synthesis_guidelines.md`（准则）配套 —— 准则定义"做到什么标准"，本文定义"怎么做到"。

---

## 1. 目标与范围

针对 7 个 Theory-of-Mind 评测任务（ToMBench / SocialIQA / BigToM / EmoBench / FanToM / HiToM / SimpleToM），从已有模型的**预测错误**出发，诊断错误原因，合成高质量训练样本。

每条合成样本的存储格式（与评测 pipeline 完全一致）：

```python
{
    "story":    str,
    "question": str,
    "answer": {"correct_answers": [...], "wrong_answers": [...]},
    "meta": {"id": "synthetic_<dataset>_iter<N>_<r>_<s>", ...},
    "data_source": "synthesized"
}
```

最终可用数据：`data_output/synth_clean/<dataset>/synthetic_iter{N}_*_hard.parquet`

---

## 2. 总体架构（5-stage pipeline）

```
tomeval_predictions_latest_full/
  <model>/<dataset>_prediction.jsonl      (已有评测结果，2+ 模型)
          │
          ▼
Stage 1: stage1_load_predictions.py
  └─ 三模型 bad case 并集
  └─ 按维度分层抽样（max_bad_cases 时保证覆盖）
  └─ 标记 _hard（≥2 模型均错）
  → data_output/bad_cases/<dataset>/bad_cases.jsonl
          │
          ▼
Stage 2: stage2_diagnosis.py
  └─ 按 get_dimension_key() 分组
  └─ 每维度 ceil(n/K) 批，每批 K 条 bad case → 合成模型
  └─ 输出 DimensionDiagnosisReport（error_patterns / cognitive_ops / synthesis_themes）
  → data_output/diagnosis_reports/<dataset>/<split>/dimension_reports.jsonl
  → data_output/diagnosis_reports/<dataset>/<split>/dimension_coverage.json
          │
          ▼
Stage 3: stage3_synthesis.py
  └─ 每份 DimensionDiagnosisReport → samples_per_report 道题
  └─ Pydantic schema 校验（per-dataset）
  └─ 最多 max_retries 轮重试（失败才重，成功不重）
  → data_output/synth_raw/<dataset>/candidates_iter{N}_<model>.jsonl
          │
          ▼
Stage 4: stage4_lsh_filter.py
  └─ 4-gram MinHash LSH（128 perm，threshold=0.6）
  └─ 命中测试集 → 丢弃，写 _dropped.jsonl + _dedupe_log.jsonl
  → data_output/synth_clean/<dataset>/synthetic_iter{N}_<model>.parquet
          │
          ▼
Stage 5: stage5_difficulty_filter.py
  └─ qwen3-8b @ DashScope 跑 repeats 次推理
  └─ 全部答对 → 丢弃（题目太简单，训练价值低）
  └─ 至少 1 次答错 → 保留
  → data_output/synth_clean/<dataset>/synthetic_iter{N}_<model>_hard.parquet ✓
```

---

## 3. 关键模块与代码位置

| 文件 | 作用 |
|------|------|
| `feedback_synthesis/stage1_load_predictions.py` | 从预测文件加载 bad case；维度分层抽样；标记 `_hard` |
| `feedback_synthesis/stage2_diagnosis.py` | 按维度分组批量诊断；`get_dimension_key()`；`allocate_batches()`；写 `dimension_coverage.json` |
| `feedback_synthesis/stage3_synthesis.py` | Pydantic schema registry；`synthesize_from_reports()`；`run_stage3_synthesis()` |
| `feedback_synthesis/stage4_lsh_filter.py` | 封装 `merge_and_dedupe` 的 LSH 守门员；写 drop/log 文件 |
| `feedback_synthesis/stage5_difficulty_filter.py` | qwen3-8b 难度验证；复用 `src/evaluation/prompts.py` 同款 prompt |
| `feedback_synthesis/prompts.py` | 诊断 prompt（`build_batch_diagnosis_prompt`）；合成 prompt（`build_stage2_generation_from_report_prompt`）；dataset schema 注册表 |
| `feedback_synthesis/config.yaml` | 所有模型 / 路径 / 阶段参数 |
| `run_data_processing.py` | 入口：解析 `--stage`，顺序调用各阶段 |
| `scripts/data/merge_and_dedupe.py` | LSH 核心实现（build_test_index / filter_candidates） |

---

## 4. Stage 1：Bad Case 加载与分层抽样

**数据来源**：`tomeval_predictions_latest_full/<model>/<dataset>_prediction.jsonl`，当前使用两个训练模型并集：
- `HardToM_GRPO_synthetic_20260515_hf`
- `HardToM_GRPO_synthetic_20260515_hf_3epoch`

**并集逻辑**：样本在任一模型下任一 repeat 答错 → 进入 bad case 池。  
**`_hard` 标记**：≥2 个模型均答错的样本，是更高价值的合成目标（诊断更能命中共性弱点）。

**维度分层抽样**（`max_bad_cases > 0` 时生效）：
1. 对每条 bad case 调用 `get_dimension_key(meta, dataset_name)` 取维度 key
2. 按维度分桶，轮询各桶取样（保证长尾维度不被截断）
3. 日志打印 `Coverage by dimension: {dim: count}` 供核查

**维度 key 映射**（`stage2_diagnosis.get_dimension_key`）：

| 数据集 | 维度字段 |
|--------|---------|
| ToMBench | `meta.ability` |
| BigToM | `meta.condition_type`（6 种） |
| SocialIQA / SimpleToM / EmoBench | `meta.dimension[0]` |
| HiToM | `meta.order` → `order_N` |
| FanToM | `meta.question_type` |

---

## 5. Stage 2：维度批量诊断

**批次分配**：`batches_c = ceil(n_c / K)`，每个维度至少 1 批，K = `samples_per_batch`（默认 5）。  
不设总批次上限——所有 bad case 都要被覆盖，不能按比例截断。

**诊断 Prompt** 输入：
- 该维度的 K 条 bad case（story + question + 正确答案 + 模型错误答案 + 模型 reasoning）
- dataset 的技能说明（`DATASET_SKILL_REGISTRY`）

**输出 schema**（`DimensionDiagnosisReport`）：

```python
{
    "dimension": str,
    "common_error_patterns": list[str],     # 用占位符 Agent A/B/Object X
    "primary_cognitive_operation": Literal[...],  # 11 种认知操作
    "secondary_cognitive_operations": list[str],
    "recommended_synthesis_themes": list[str],
    "difficulty_distribution": {"easy": float, "medium": float, "hard": float}
}
```

**兜底策略**：LLM 解析失败时使用 `_FALLBACK_REPORT`，标记 `_is_fallback=True`，不阻塞后续流程。

**可观测产物**：`dimension_coverage.json` 记录 `{dim_counts, allocations, n_batches}`，供迭代日志引用。

---

## 6. Stage 3：从诊断报告合成新样本

**合成逻辑**：每份 `DimensionDiagnosisReport` → `samples_per_report` 道题（默认 3）。  
生成 prompt 包含：error patterns、synthesis themes、difficulty distribution、dataset 格式规范。

**重试机制**：
- 第一轮：所有 (report, sample_idx) 对并行生成
- 生成失败（LLM 返回 null / schema 校验不通过）→ 进入重试队列
- 最多 `max_retries` 轮（默认 3），重试时附上上一次失败的输出供 LLM 参考
- 成功即出队，不重复生成

**per-dataset Pydantic schema**（`SYNTHESIS_SCHEMA_REGISTRY`）：

| 数据集 | Schema | 核心约束 |
|--------|--------|---------|
| ToMBench | `ToMBenchSynthesisOutput` | 4选项嵌入问题文本 |
| SocialIQA | `SocialIQASynthesisOutput` | 3选项 A/B/C |
| BigToM | `BigToMSynthesisOutput` | 2选项 A/B，需有 condition_type |
| EmoBench | `EmoBenchSynthesisOutput` | 变长文本选项 |
| FanToM | `FanToMSynthesisOutput` | 含 question_type + story_idx |
| HiToM | `HiToMSynthesisOutput` | 15 选项 A–O，有 order 字段 |
| SimpleToM | `SimpleToMSynthesisOutput` | 含固定 Empty 选项 |

**meta.id 格式**：`synthetic_<dataset>_iter{N}_{r:03d}_{s}`（全局唯一，防下游 ID 冲突）

---

## 7. Stage 4：LSH 守门员过滤

**目的**：防止合成样本在 story+question+answers 层面与测试集相似，保证评测有效性。

**实现**：`scripts/data/merge_and_dedupe.py:filter_candidates`
- 4-gram MinHash，128 perm，Jaccard threshold = 0.6
- 输入：`synth_raw/<dataset>/candidates_iter{N}_*.jsonl`
- 通过 → `synth_clean/*.parquet`
- 拦截 → `*_dropped.jsonl` + `*_dedupe_log.jsonl`（含 `_drop_reason`）

**配置**（`config.yaml:leakage_guard`）：
```yaml
test_root: ../test_data
threshold: 0.6
num_perm: 128
ngram: 4
```

---

## 8. Stage 5：难度验证（qwen3-8b）

**目的**：弱模型能轻易答对的题训练价值低，丢弃。

**实现**：复用 `src/evaluation/prompts.py` 的同款 prompt（`build_option_bundle` 确定性 shuffle + `build_prompt`），与正式评测语义完全一致。

**决策规则**（每条样本跑 `repeats` 次，默认 3）：
- 至少 1 次答错 → **保留**，写入 `*_hard.parquet`
- 全部答对 → 丢弃（计入 `dropped_all_correct`）
- API 全部失败 → 默认丢弃（`keep_when_api_fail=false`，保守策略）

**弱模型配置**：`qwen3-8b @ DashScope`，`enable_thinking=false`，`max_tokens=4096`

---

## 9. 入口命令参考

```bash
source /Users/yangmeili/Downloads/Code/.venv/bin/activate
cd /Users/yangmeili/Downloads/Code/ToMEval

# 完整流水线
python run_data_processing.py --stage all --dataset ToMBench --iteration 1

# 分阶段调试
python run_data_processing.py --stage load     --dataset BigToM --max-bad-cases 80
python run_data_processing.py --stage diagnose --dataset BigToM
python run_data_processing.py --stage synth    --dataset BigToM --iteration 1
python run_data_processing.py --stage difficulty --dataset BigToM --iteration 1

# 全部数据集一次性跑
python run_data_processing.py --stage all --iteration 2
```

跑完核查：
1. `data_output/diagnosis_reports/<dataset>/*/dimension_coverage.json` — 确认各维度都有覆盖
2. `data_output/synth_clean/<dataset>/synthetic_iter{N}_*_hard.parquet` — 最终可用数据行数
3. `feedback_synthesis/ITERATION_LOG.md` — 各阶段统计摘要

**可接受 kept_rate 范围**：60%–95%。低于 30% 说明合成模型造题过于简单，需加强诊断 prompt 的难度提示。

---

## 10. 配置文件关键参数速查

```yaml
synthesis_model:
  model_name: deepseek-v4-flash   # 强模型负责诊断和合成
  temperature: 0.8
  max_workers: 16

synthesis:
  samples_per_batch: 5            # Stage 2 每批 bad case 数 K
  samples_per_report: 3           # Stage 3 每份报告生成几道题
  max_retries_per_diagnosis: 3    # Stage 3 生成失败最多重试次数

difficulty_filter:
  model_name: qwen3-8b            # 弱模型负责难度验证
  repeats: 3                      # 每题推理次数
  keep_when_api_fail: false       # API 全失败时默认丢弃
```
