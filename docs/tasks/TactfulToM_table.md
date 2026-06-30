# TactfulToM — 指标定义

- 指标层级与 `metrics.json` 的 `avg_metrics.dimensions` 树一一对应：
  **一级**=总体 accuracy；**二级**=各维度；**三级 / 四级**=维度内嵌套子维度。
- `切分型`：把数据集切成多个 split，各 split 一条准确率；`汇总型`：任务特有口径（如配对联合、宏平均、set 级 ALL），单值无法从边际准确率反推。
- 本页只列指标定义，不含具体数值。

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
| `accuracy` | `category` | — | — |
|  | `joint_comp_just` | — | — |
|  | `lie_type` | — | — |
|  | `question_type` | — | — |
|  | `tom_type` | — | — |
|  | `type` | — | — |

## 各维度定义

### 二级指标 · `category`（切分型，单位 ACC）

按问题大类切分（meta.category）：belief / answerability / info_accessibility / lieability / liedetectability / justification。

### 二级指标 · `joint_comp_just`（汇总型，单位 ACC）

Comp∧Just 联合指标（汇总型，单 overall split）。同一对话的 Comprehension 与 Justification 都答对才算真正理解了善意谎言（Happé 双题判定）。

### 二级指标 · `lie_type`（切分型，单位 ACC）

按白谎类型切分（meta.lie_type）：altruistic_white_lies / pareto_white_lies。

### 二级指标 · `question_type`（切分型，单位 ACC）

按细粒度题型切分（meta.question_type），如 tom:belief:accessible:truth 等完整路径。

### 二级指标 · `tom_type`（切分型，单位 ACC）

按 ToM 阶数与角色组合切分（meta.tom_type）：first-order / second-order 各角色对。

### 二级指标 · `type`（切分型，单位 ACC）

题型维度（所有数据集固定带）。按 prompt_type 切分：mcq_single（单选）/ mcq_multi（多选）/ mcq_grouped（捆绑判分）/ open（开放题）。每个 split 是该题型的准确率。
