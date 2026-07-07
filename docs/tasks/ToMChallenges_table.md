# ToMChallenges — 指标定义

- 指标层级与 `metrics.json` 的 `avg_metrics.dimensions` 树一一对应：
  **一级**=总体 accuracy；**二级**=各维度；**三级 / 四级**=维度内嵌套子维度。
- `切分型`：把数据集切成多个 split，各 split 一条准确率；`汇总型`：任务特有口径（如配对联合、宏平均、set 级 ALL），单值无法从边际准确率反推。
- 本页只列指标定义与逐值释义，不含具体数值。
- 全中文版见 [ToMChallenges_table_zh.md](ToMChallenges_table_zh.md)。

## 一级指标

| 指标 | 定义 |
|---|---|
| accuracy | 一级指标。全部样本的总体准确率 = correct / total。open 题按各数据集判分模式（f1 / rubric）二值化后计入。 |
| correct | 答对样本数。 |
| total | 参与评测的样本总数。 |
| extraction_failed | 答案抽取失败数（MCQ 严格模式下未输出 \boxed{} 即记为抽取失败）。 |
| extraction_failed_rate | 抽取失败率 = extraction_failed / total，衡量模型对输出格式的遵循程度。 |

## 指标层级总表

> 横轴为指标层级；同一行表示嵌套归属（如 `dim1 → dim2 → dim3` 表示 dim2 是 dim1 的子维度、dim3 又是 dim2 的子维度）。`—` 表示该维度没有更深层级。

| 一级指标 | 二级指标 | 三级指标 | 四级指标 |
|---|---|---|---|
| `accuracy` | `type` | — | — |
|  | `question_type` | — | — |
|  | `task_format` | — | — |
|  | `test_type` | — | — |

## 各维度定义

### 二级指标 · `type`（切分型，单位 ACC）

题型维度（所有数据集固定带）。按 prompt_type 切分，每个 split 是该题型的准确率。

- `mcq_single`：仅一个正确项 + 干扰项的选择题
- `mcq_multi`：有 ≥2 个正确项的选择题
- `mcq_grouped`：一条 prompt 内含多个子问题，全部答对才计为对（如 EmoBench EU 的情绪+原因）
- `open`：无干扰项、自由作答，按 f1 / rubric 判分后二值化

### 二级指标 · `question_type`（切分型，单位 ACC）

按题型切分（meta.question_type）。

- `1stA`：一阶信念题（问法 A）
- `1stB`：一阶信念题（问法 B）
- `2ndA`：二阶信念题（问法 A）
- `2ndB`：二阶信念题（问法 B）
- `assumption`：对前提假设的判断
- `memory`：记忆核对题
- `reality`：现实核对题

### 二级指标 · `task_format`（切分型，单位 ACC）

按作答形式切分（meta.task_format）。

- `mc`：多选一形式
- `qa`：开放问答形式

### 二级指标 · `test_type`（切分型，单位 ACC）

按经典 ToM 测试范式切分（meta.test_type）。

- `sally-anne`：意外转移范式
- `smarties`：意外内容范式
