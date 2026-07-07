# Repair 模块

数据自动修复模块，包含 V3 数据飞轮的 Phase E 修复逻辑。

## 模块组成

### repair_pipeline.py - 修复编排器

**功能**：根据评估结果自动修复问题样本，生成下一轮迭代输入。

**输入**：
- `input_df`: 原始样本 DataFrame
- `labels_df`: 评估标签 DataFrame（含 `label` 和 `repair_type` 字段）

**输出**：
- `repaired.parquet`: 修复成功的样本（下一轮输入）

**修复类型**：

| repair_type | 触发条件 | 修复策略 |
|------------|---------|---------|
| `easy` | Phase B 全部通过（all_passed） | 增加难度：添加干扰选项、复杂化问题、引入多步推理 |
| `unanswerable` | Phase C 判定为不可回答 | 补充信息：添加缺失的故事细节、修正逻辑矛盾 |
| `shortcut` | Phase D 检测到捷径 | 消除捷径：根据具体维度（no_story/no_question/no_options）针对性修复 |

**修复流程**：
1. 筛选需要修复的样本（`repair_type` 非空且在 `enabled_types` 中）
2. 按 `repair_type` 分组
3. 调用 LLM 生成修复后的样本（使用 `StructureClient` 确保结构化输出）
4. 验证修复结果（schema 校验、必填字段检查）
5. 返回修复成功的样本

**关键函数**：
- `repair_samples()`: 主入口，编排整个修复流程
- `_repair_batch()`: 批量调用 LLM 修复
- `_validate_repaired()`: 验证修复结果
- `write_repaired_parquet()`: 写入输出文件

---

### repair_prompts.py - 修复提示词模板

**功能**：为不同修复类型提供专门的提示词模板。

**模板类型**：

#### 1. `REPAIR_EASY_PROMPT`
针对过于简单的样本（all_passed）。

**修复策略**：
- 添加更多干扰选项（相似但错误的答案）
- 增加推理步骤（需要多步思考）
- 引入隐含信息（需要推断而非直接陈述）
- 复杂化问题表述

**示例**：
```
原始：Tom 把球放在盒子里。球在哪里？
修复：Tom 把球放在盒子里，然后离开了房间。Mary 进来后把盒子移到了桌子下面。Tom 回来时，他会在哪里找球？
```

#### 2. `REPAIR_UNANSWERABLE_PROMPT`
针对不可回答的样本（answerable=False）。

**修复策略**：
- 补充缺失的关键信息
- 修正逻辑矛盾
- 明确模糊的指代
- 确保问题与故事匹配

**示例**：
```
原始：Tom 很高兴。他为什么高兴？（故事中未提及原因）
修复：Tom 收到了期待已久的礼物，他很高兴。他为什么高兴？
```

#### 3. `REPAIR_SHORTCUT_PROMPT`
针对存在捷径的样本（is_shortcut=True）。

**修复策略**（根据检测到的维度）：
- **no_story 捷径**：确保答案依赖故事而非常识
- **no_question 捷径**：确保选项不过于明显
- **no_options 捷径**：确保问题本身不暗示答案

**示例**：
```
原始（no_story 捷径）：
  故事：Tom 去了商店。
  问题：太阳从哪边升起？
  选项：A. 东边  B. 西边
  
修复：
  故事：Tom 早上醒来，看到阳光从卧室的东窗照进来。
  问题：根据故事，Tom 的卧室窗户朝哪个方向？
  选项：A. 东边  B. 西边
```

**提示词结构**：
```python
{
    "system": "你是数据质量专家...",
    "user": """
原始样本：
{story}
{question}
{options}

问题类型：{repair_type}
诊断信息：{diagnostic_info}

请修复该样本，确保：
1. [具体要求]
2. [具体要求]
...

输出格式：
{
  "story": "修复后的故事",
  "question": "修复后的问题",
  "answer": {
    "correct_answers": [...],
    "wrong_answers": [...]
  },
  "meta": {...}
}
"""
}
```

---

## 使用示例

```python
from filter.repair import repair_samples
from filter.base import load_judge_client
import pandas as pd

# 加载数据
input_df = pd.read_parquet("synthetic.parquet")
labels_df = pd.read_parquet("eval_iter1/labels.parquet")

# 加载修复客户端
repair_client = load_judge_client("strong")

# 执行修复
repaired_df = repair_samples(
    df=input_df,
    labels_df=labels_df,
    dataset="MyDataset",
    iter_n=1,
    repair_client=repair_client,
    enabled_types=["easy", "unanswerable", "shortcut"],
)

# 保存结果
repaired_df.to_parquet("eval_iter1/repaired.parquet", index=False)
```

---

## 修复质量保证

### 1. 结构化输出
使用 `StructureClient` 强制 LLM 输出符合 schema 的 JSON，避免解析错误。

### 2. 字段验证
- 必填字段：`story`, `question`, `answer.correct_answers`, `answer.wrong_answers`
- 类型检查：确保 `correct_answers` 和 `wrong_answers` 是列表
- 非空检查：`story` 和 `question` 不能为空字符串

### 3. 失败处理
- 修复失败的样本不会进入下一轮
- 记录失败原因到日志
- 最后一轮仍失败的样本标记为 `unfixable`

### 4. 迭代上限
- 默认最多 2 轮修复（含第一轮，可配置 `max_iter`）
- 避免无限循环修复

---

## 输出文件格式

`repaired.parquet` 包含以下字段：
- `story`: 修复后的故事
- `question`: 修复后的问题
- `answer`: 修复后的答案对象
  - `correct_answers`: 正确答案列表
  - `wrong_answers`: 错误答案列表
- `meta`: 元数据（保留原始 `id`，添加 `repaired_from_iter` 等）

---

## 设计原则

1. **类型驱动**：根据 `repair_type` 选择不同修复策略
2. **上下文感知**：修复时提供诊断信息（如 shortcut 的具体维度）
3. **保守修复**：只修复明确需要修复的样本，避免过度修改
4. **可追溯性**：在 `meta` 中记录修复历史

---

## 相关文档

- 评估流程可视化：`../OVERVIEW.md`
- 评估模块：`../eval/README.md`
- 决策树编排：`../run_filter.py`
