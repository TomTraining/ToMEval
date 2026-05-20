# 数据合成准则（synthesis_guidelines.md）

本文件是合成数据迭代的"宪法"。每轮合成结束后，对照这 4 条准则核查产出；任一 FAIL 必须按下方"FAIL 行动"调整参数/prompt/配置后再进入下一轮。

> 配套方案：`data_processing/synthesis_plan.md`

---

## 准则 A — 维度全覆盖

合成数据必须覆盖测试集中出现的**所有能力维度**。Stage 1 加载 bad case 时不能让长尾维度被 `max_bad_cases` 截断。

**量化指标**

- Stage 2 输出的 `dimension_coverage.json` 中，每个数据集的 `dim_counts` 键集合必须覆盖该数据集测试集的所有主要维度
- 对 BigToM，6 种 `condition_type` 全部出现：`forward_action / backward_belief / true_belief / percept_to_belief / forward_belief / false_belief`
- 对 HiToM，各 order（1/2/3/4）均须有 bad case 进入诊断批次
- 对 FanToM，`beliefQAs` 和 `infoAccessibilityQAs` 两类均须覆盖

**实现保障**

Stage 1 通过 `get_dimension_key()` 分桶后轮询取样（而非直接截断），保证每个维度至少有 1 条进入诊断。

**FAIL 行动**

1. 检查 `dimension_coverage.json`，找出缺失维度
2. 降低 `--max-bad-cases` 或增大总 bad case 数，使稀少维度的样本能排上
3. 若某维度测试集本身样本极少（< 5 条），可豁免

---

## 准则 B — 难度有保障

合成数据对弱模型（qwen3-8b）不能过于简单。qwen3-8b 全部答对的样本对 GRPO 训练价值低，必须丢弃。

**量化指标**

- Stage 5 的 `kept_rate`（难度通过率）应在 **60%–95%**
  - `< 30%`：合成模型造题整体过于简单，合成质量不达标
  - `> 98%`：几乎所有题弱模型都答错，可能干扰项质量有问题或题目本身有缺陷
- `dropped_all_correct`（全对被丢）/ `total` ≤ 70%（即 kept_rate ≥ 30%）

**当前实现**

Stage 5 对每条样本跑 `repeats=3` 次推理，使用与正式评测相同的 prompt（确定性 shuffle）。API 失败时默认丢弃（保守策略，`keep_when_api_fail=false`）。

**FAIL 行动**

若 `kept_rate < 30%`：
1. 检查 Stage 3 的合成 prompt，在 `build_stage2_generation_from_report_prompt` 中加强难度引导（如："干扰项需要 ≥2 步推理才能排除"）
2. 检查 `difficulty_distribution` 字段——若诊断报告里 `easy` 比例偏高，可在 Stage 2 prompt 中要求诊断报告更多关注中难度/高难度错误模式
3. 确认 `wrong_answers` 字段不为空（空干扰项会导致题目退化为开放题）

若 `kept_rate > 98%`：
1. 人工抽查 5 条被弱模型答错的样本，确认正确答案是否真正合理
2. 检查 option shuffle 逻辑是否正常（`build_option_bundle` 的 sample_id 是否唯一）

---

## 准则 C — 零数据泄露

合成样本严禁与 `test_data/` 中任何样本在 story+question+answers 层面高度相似，包括"换名字结构抄袭"。

**量化指标**

- Stage 4 通过 LSH 守门员（threshold=0.6，4-gram MinHash 128 perm）的样本 `synth_clean/*.parquet` 即为无泄露样本
- `*_dedupe_log.jsonl` 中 `drop_rate` 为可接受的信息；如果 `drop_rate > 50%` 需要人工核查合成 prompt 是否"抄袭"了测试集 story 结构

**实现保障**

Stage 4 在写入 `synth_clean` 之前强制过 LSH 守门员，任何未经 Stage 4 的 `synth_raw` 数据**不得直接使用**。最终训练数据**只使用** `*_hard.parquet`。

**FAIL 行动**

若 `drop_rate > 50%`：
1. 检查 `*_dropped.jsonl` 中被丢弃样本的 `_drop_reason`，看是否集中在某个数据集的某个维度
2. 若某维度的合成 prompt 模板导致大量相似 story 结构，修改 `build_stage2_generation_from_report_prompt` 中对应 dataset 的格式规范，要求多样化场景背景
3. 不能通过降低 LSH 阈值（threshold）来"解决"泄露率高的问题

---

## 准则 D — 配比合理

弱维度多合成，已有大量训练数据的数据集少合成；合成数量要与当前模型的错误分布匹配。

**量化指标**

- Stage 1 加载的 bad case 分布（`dimension_coverage.json` 的 `dim_counts`）应与测试集错误分布一致
- 各数据集每轮合成量建议参考模型错误率（`_hard` 比例高的数据集优先多合成）：
  - `_hard`（≥2 模型均错）比例 > 40% → 该数据集弱点明显，多分配合成预算
  - `_hard` 比例 < 10% → 已基本掌握，少分配或暂跳过

**FAIL 行动**

若某数据集 `_hard` 比例异常低但模型整体 accuracy 也不高：
1. 检查两个训练模型的预测文件是否都存在（`DATASET_FILE_MAP` 中的文件名是否匹配）
2. 若某模型文件缺失，Stage 1 会打 warning，此时 bad case 退化为单模型，`_hard` 自然为 0

---

## 每轮闭环

```
[轮次 N]
  python run_data_processing.py --stage all --iteration N
        │
        ├─ Stage 1: 加载 bad case，输出 bad_cases.jsonl
        ├─ Stage 2: 维度诊断，输出 dimension_reports.jsonl + dimension_coverage.json
        ├─ Stage 3: 合成，输出 candidates_iter{N}_*.jsonl
        ├─ Stage 4: LSH 过滤，输出 synthetic_iter{N}_*.parquet
        └─ Stage 5: 难度验证，输出 synthetic_iter{N}_*_hard.parquet
                │
                ▼
        核查 ITERATION_LOG.md 末尾统计段
        检查 dimension_coverage.json      ← 准则 A
        检查 kept_rate                    ← 准则 B
        检查 drop_rate in dedupe_log      ← 准则 C
        检查 _hard 比例分布               ← 准则 D
                │
                ▼
        PASS → 进入轮次 N+1 或放量生产
        FAIL → 按上方行动清单调整后重跑
```

---

## PASS 标准汇总

| 准则 | PASS 条件 |
|------|-----------|
| A 维度覆盖 | `dimension_coverage.json` 无缺失关键维度；BigToM 6 种 condition_type 全到；FanToM 两类 question_type 均有 |
| B 难度 | `kept_rate ∈ [30%, 98%]`，建议 60%–95% |
| C 泄露 | `synth_clean/*.parquet` 全部经过 Stage 4 LSH；`drop_rate ≤ 50%`（否则检查合成 prompt） |
| D 配比 | `_hard` 高比例数据集获得更多合成预算；各数据集 bad case 分布与测试集错误分布一致 |

---

## 关键路径约束（必须遵守）

1. **`synth_raw` ≠ 训练数据**：任何 `synth_raw/` 下的文件必须经过 Stage 4（LSH）和 Stage 5（难度）才能使用
2. **只使用 `*_hard.parquet`**：`synth_clean/` 下不带 `_hard` 后缀的 parquet 是 Stage 4 中间产物
3. **合成模型 ≠ 难度验证模型**：合成用强模型（deepseek-v4-flash），难度验证用弱模型（qwen3-8b），两者不可混用
4. **维度分层早于 max_bad_cases 截断**：Stage 1 必须先按维度分桶再截断，不能直接按数量截断
