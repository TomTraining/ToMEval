# Eval 模块

数据质量评估模块，包含 V3 数据飞轮的 Phase B/C/D 三个评估阶段。

## 模块组成

### eval_passk.py - Phase B: Pass@k 评估

**功能**：使用弱模型对样本进行 k 次独立预测，根据通过率分桶。

**输入**：标准化数据集 DataFrame（含 story/question/answer 字段）

**输出**：
- `passk.parquet`：每条样本的 pass@k 结果
  - `sample_id`: 样本 ID
  - `pass_at_k`: 通过次数（0-k）
  - `bucket`: 分桶结果（`all_passed` / `partial` / `all_failed`）

**分桶规则**：
- `all_passed`: k 次全部正确 → 样本过于简单
- `all_failed`: k 次全部错误 → 样本可能无法回答或过难
- `partial`: 部分正确 → 需进一步检测

**关键参数**：
- `k=3`: 默认尝试 3 次
- `simple_client`: 弱模型客户端（如 Qwen3-8B）

---

### eval_answerability_full.py - Phase C: 可回答性判断

**功能**：使用强模型判断样本是否可回答（是否有足够信息得出答案）。

**输入**：Phase B 中 `partial` 或 `all_failed` 的样本

**输出**：
- `answerability.parquet`：每条样本的可回答性判断
  - `sample_id`: 样本 ID
  - `answerable`: 布尔值（True/False）
  - `label`: 判断标签（`answerable` / `unanswerable`）
  - `reason`: 判断理由

**判断标准**：
- `answerable=True`: 故事提供了足够信息，理论上可以回答
- `answerable=False`: 信息不足、逻辑矛盾或问题本身有问题

**关键参数**：
- `strong_client`: 强模型客户端（如 Claude-Opus）

---

### eval_shortcut.py - Phase D: 捷径检测

**功能**：三维度检测样本是否存在捷径（不需要完整信息就能答对）。

**输入**：Phase B 中 `partial` 且 Phase C 判定为 `answerable` 的样本

**输出**：
- `shortcut.parquet`：每条样本的捷径检测结果
  - `sample_id`: 样本 ID
  - `no_story_pass`: 无故事时的通过次数
  - `no_question_pass`: 无问题时的通过次数
  - `no_options_pass`: 无选项时的通过次数（仅 MCQ）
  - `is_shortcut`: 布尔值（True/False）

**三个维度**：
1. **no_story**: 移除故事，仅保留问题和选项 → 测试是否依赖常识而非故事
2. **no_question**: 移除问题，仅保留故事和选项 → 测试选项是否过于明显
3. **no_options**: 移除选项，仅保留故事和问题 → 测试问题本身是否暗示答案

**判定规则**：
- 任一维度通过次数 ≥ `threshold`（默认 `majority = ceil(k/2)`）→ `is_shortcut=True`
- 所有维度通过次数 < `threshold` → `is_shortcut=False`

**关键参数**：
- `k=3`: 每个维度尝试 3 次
- `threshold="majority"`: 阈值为多数（2/3）
- `dimensions`: 启用的检测维度列表
- `simple_client`: 弱模型客户端
- `judge_client`: 判题客户端（用于 no_options 维度的开放式回答判题）

---

## 使用示例

```python
from filter.eval import run_passk_on_df, run_answerability_on_df, run_shortcut_on_df
from filter.base import load_answer_models, load_judge_client
import pandas as pd

# 加载数据
df = pd.read_parquet("synthetic.parquet")

# 加载客户端
clients = load_answer_models()
simple_client = clients["simple"]
strong_client = clients["strong"]
judge_client = load_judge_client("strong")

# Phase B: pass@k
passk_df = run_passk_on_df(df, dataset="MyDataset", k=3, simple_client=simple_client)

# Phase C: answerability（仅 partial + all_failed）
ans_target = df[passk_df["bucket"].isin(["partial", "all_failed"])]
ans_df = run_answerability_on_df(ans_target, dataset="MyDataset", strong_client=strong_client)

# Phase D: shortcut（仅 partial + answerable）
partial_ids = set(passk_df[passk_df["bucket"] == "partial"]["sample_id"])
answerable_ids = set(ans_df[ans_df["answerable"] == True]["sample_id"])
sc_target = df[df["meta"].apply(lambda m: m.get("id") in (partial_ids & answerable_ids))]
shortcut_df = run_shortcut_on_df(
    sc_target,
    dataset="MyDataset",
    k=3,
    threshold="majority",
    simple_client=simple_client,
    judge_client=judge_client,
)
```

---

## 输出文件格式

所有输出均为 Parquet 格式，包含以下公共字段：
- `sample_id`: 样本唯一标识（从 `meta.id` 提取）
- 各阶段特定字段（见上文各模块说明）

---

## 设计原则

1. **分阶段过滤**：逐步缩小检测范围，避免对所有样本执行昂贵操作
2. **决策树驱动**：Phase B 分桶 → Phase C 判断 → Phase D 检测，每阶段结果决定下一阶段输入
3. **可复现性**：所有随机操作使用固定种子，确保结果可复现
4. **模块化**：每个阶段独立运行，输出独立文件，便于调试和增量更新

---

## 相关文档

- 评估流程可视化：`../OVERVIEW.md`
- 决策树编排：`../run_filter.py`
- 修复模块：`../repair/README.md`
