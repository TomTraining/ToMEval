# SocialMind — 指标定义

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
| `accuracy` | `dim1` | `dim2` | `dim3` |
|  | `length` | — | — |
|  | `perspective` | — | — |
|  | `q4_score` | — | — |
|  | `qtype` | — | — |
|  | `type` | — | — |
|  | `variant` | — | — |

## 各维度定义

### 二级指标 · `dim1`（切分型，单位 ACC）

SocMind 维度体系一级（meta.dim1），其下嵌套 dim2（二级标签，三级维度）→ dim3（三级标签，四级维度）。1/2/3 为顶层能力大类。

### 三级指标 · `dim1 → dim2`（切分型，单位 ACC）

维度体系二级标签（如 1.1 / 2.4，三级维度，挂在 dim1 各 split 下）。

### 四级指标 · `dim1 → dim2 → dim3`（切分型，单位 ACC）

维度体系三级标签（如 1.1.2 / 2.4.7，四级维度，挂在 dim2 各 split 下），是 SocMind 最细粒度的考察点。

### 二级指标 · `length`（切分型，单位 ACC）

按情景文本长短切分（meta.length_mode）：long / short。

### 二级指标 · `perspective`（切分型，单位 ACC）

按叙事视角切分（meta.perspective）：first_person（第一人称）/ third_person（第三人称）。

### 二级指标 · `q4_score`（汇总型，单位 0-10 均分）

Q4 rubric 平均分（汇总型，0–10 分而非 0–1 准确率）。overall=全部 Q4 均分，其余 split 按三级维度 meta.dim 分组。

### 二级指标 · `qtype`（切分型，单位 ACC）

按题型编号切分（meta.qtype）：Q1 / Q2 / Q3 为客观题（单/多选），Q4 为开放分析题（走 rubric LLM 判分，0–10 分过阈值）。

### 二级指标 · `type`（切分型，单位 ACC）

题型维度（所有数据集固定带）。按 prompt_type 切分：mcq_single（单选）/ mcq_multi（多选）/ mcq_grouped（捆绑判分）/ open（开放题）。每个 split 是该题型的准确率。

### 二级指标 · `variant`（切分型，单位 ACC）

按难度变体切分（meta.variant）：base（全量基础题）/ hardest（dim-3 加难版）/ varA / varB（其他改写变体）。

## 人工审核与 qualified 镜像

| 指标 | 定义 |
|---|---|
| review_pass_count | 人工审核合格（meta.review_pass=True）样本数。 |
| review_fail_count | 审核不合格样本数。 |
| review_pass_rate | 合格率 = review_pass_count / total。 |

> `qualified` 是一份镜像：仅在审核合格样本上**重算同一套**一级/二级/三级/四级指标，结构与上文完全一致（accuracy + dimensions 树）。v5.3 默认全部合格时，qualified 与全量一致。
