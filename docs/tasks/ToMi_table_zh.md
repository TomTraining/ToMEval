# ToMi — 指标定义（全中文版）

- 指标层级与 `metrics.json` 的 `avg_metrics.dimensions` 树一一对应：
  **一级**=总体 accuracy；**二级**=各维度；**三级 / 四级**=维度内嵌套子维度。
- `切分型`：把数据集切成多个 split，各 split 一条准确率；`汇总型`：任务特有口径（如配对联合、宏平均、set 级 ALL），单值无法从边际准确率反推。
- 本页只列指标定义与逐值释义，不含具体数值。
- 本页为全中文版：指标名 / 字段值一律中文（英文标识符括注在后）。英文技术版见 [ToMi_table.md](ToMi_table.md)。

## 一级指标

| 指标 | 定义 |
|---|---|
| 准确率（accuracy） | 一级指标。全部样本的总体准确率 = correct / total。open 题按各数据集判分模式（f1 / rubric）二值化后计入。 |
| 答对数（correct） | 答对样本数。 |
| 样本总数（total） | 参与评测的样本总数。 |
| 抽取失败数（extraction_failed） | 答案抽取失败数（MCQ 严格模式下未输出 \boxed{} 即记为抽取失败）。 |
| 抽取失败率（extraction_failed_rate） | 抽取失败率 = extraction_failed / total，衡量模型对输出格式的遵循程度。 |

## 指标层级总表

> 横轴为指标层级；同一行表示嵌套归属（如 `dim1 → dim2 → dim3` 表示 dim2 是 dim1 的子维度、dim3 又是 dim2 的子维度）。`—` 表示该维度没有更深层级。

| 一级指标 | 二级指标 | 三级指标 | 四级指标 |
|---|---|---|---|
| `accuracy` | `type` | — | — |
|  | `story_type` | — | — |
|  | `question_type` | — | — |

## 各维度定义

### 二级指标 · 题型（type）（切分型，单位 ACC）

题型维度（所有数据集固定带）。按 prompt_type 切分，每个 split 是该题型的准确率。

- 单选（`mcq_single`）：仅一个正确项 + 干扰项的选择题
- 多选（`mcq_multi`）：有 ≥2 个正确项的选择题
- 捆绑判分（`mcq_grouped`）：一条 prompt 内含多个子问题，全部答对才计为对（如 EmoBench EU 的情绪+原因）
- 开放题（`open`）：无干扰项、自由作答，按 f1 / rubric 判分后二值化

### 二级指标 · 故事型（story_type）（切分型，单位 ACC）

按故事类型切分（meta.story_type）。

- 真信念（`true_belief`）：真信念故事
- 错误信念（`false_belief`）：一阶错误信念故事
- 二阶错误信念（`second_order_false_belief`）：二阶错误信念故事
- 未标注（`unknown`）：meta.story_type 为空的样本

### 二级指标 · 题型（question_type）（切分型，单位 ACC）

按题型切分（meta.question_type）。

- 一阶-需ToM(0)（`first_order_0_tom`）：一阶、需要 ToM（变体 0）
- 一阶-需ToM(1)（`first_order_1_tom`）：一阶、需要 ToM（变体 1）
- 一阶-无需ToM(0)（`first_order_0_no_tom`）：一阶、无需 ToM（变体 0）
- 一阶-无需ToM(1)（`first_order_1_no_tom`）：一阶、无需 ToM（变体 1）
- 二阶-需ToM(0)（`second_order_0_tom`）：二阶、需要 ToM（变体 0）
- 二阶-需ToM(1)（`second_order_1_tom`）：二阶、需要 ToM（变体 1）
- 二阶-无需ToM(0)（`second_order_0_no_tom`）：二阶、无需 ToM（变体 0）
- 二阶-无需ToM(1)（`second_order_1_no_tom`）：二阶、无需 ToM（变体 1）
- 记忆（`memory`）：记忆核对题
- 现实（`reality`）：现实核对题
- 未标注（`unknown`）：meta.question_type 为空的样本
