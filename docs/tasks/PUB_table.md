# PUB — 指标定义

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
| `accuracy` | `difficulty` | — | — |
|  | `dimension` | — | — |
|  | `ethics_category` | — | — |
|  | `option_count` | — | — |
|  | `source` | — | — |
|  | `task_type` | — | — |
|  | `type` | — | — |

## 各维度定义

### 二级指标 · `difficulty`（切分型，单位 ACC）

按难度切分（meta.difficulty）。

### 二级指标 · `dimension`（切分型，单位 ACC）

按考察维度切分（meta.dimension），PUB 为 pragmatics（语用）。

### 二级指标 · `ethics_category`（切分型，单位 ACC）

按伦理类别切分（meta.ethics_category）。

### 二级指标 · `option_count`（切分型，单位 ACC）

按选项个数切分（len(options)），观察候选数对正确率的影响。

### 二级指标 · `source`（切分型，单位 ACC）

按原始来源切分（meta.dataset_source）。

### 二级指标 · `task_type`（切分型，单位 ACC）

按任务类型切分（meta.task_type）。

### 二级指标 · `type`（切分型，单位 ACC）

题型维度（所有数据集固定带）。按 prompt_type 切分：mcq_single（单选）/ mcq_multi（多选）/ mcq_grouped（捆绑判分）/ open（开放题）。每个 split 是该题型的准确率。
