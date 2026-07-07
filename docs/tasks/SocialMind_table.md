# SocialMind — 指标定义

- 指标层级与 `metrics.json` 的 `avg_metrics.dimensions` 树一一对应：
  **一级**=总体 accuracy；**二级**=各维度；**三级 / 四级**=维度内嵌套子维度。
- `切分型`：把数据集切成多个 split，各 split 一条准确率；`汇总型`：任务特有口径（如配对联合、宏平均、set 级 ALL），单值无法从边际准确率反推。
- 本页只列指标定义与逐值释义，不含具体数值。
- 全中文版见 [SocialMind_table_zh.md](SocialMind_table_zh.md)。

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
|  | `dim1` | `dim2` | `dim3` |
|  | `qtype` | — | — |
|  | `perspective` | — | — |
|  | `variant` | — | — |
|  | `length` | — | — |
|  | `q4_score` | — | — |

## 各维度定义

### 二级指标 · `type`（切分型，单位 ACC）

题型维度（所有数据集固定带）。按 prompt_type 切分，每个 split 是该题型的准确率。

- `mcq_single`：仅一个正确项 + 干扰项的选择题
- `mcq_multi`：有 ≥2 个正确项的选择题
- `mcq_grouped`：一条 prompt 内含多个子问题，全部答对才计为对（如 EmoBench EU 的情绪+原因）
- `open`：无干扰项、自由作答，按 f1 / rubric 判分后二值化

### 二级指标 · `dim1`（切分型，单位 ACC）

维度体系顶层能力大类（meta.dim1），其下嵌套 dim2 → dim3。

取值 1 / 2 / 3，为三个顶层能力大类（编号制）。

### 三级指标 · `dim1 → dim2`（切分型，单位 ACC）

维度体系二级标签（meta.dim2），挂在各 dim1 split 下。

形如 1.1 / 2.4 / 3.2，共 17 个二级标签。

### 四级指标 · `dim1 → dim2 → dim3`（切分型，单位 ACC）

维度体系三级最细标签（meta.dim），挂在各 dim2 split 下，是 SocMind 最细粒度考察点。

形如 1.1.2 / 2.4.7，共 71 个三级标签。

### 二级指标 · `qtype`（切分型，单位 ACC）

按题型编号切分（meta.qtype）。

- `Q1`：客观题（单/多选）
- `Q2`：客观题（单/多选）
- `Q3`：客观题（单/多选）
- `Q4`：开放分析题，走 rubric LLM 判分（0–10 分过阈值）

### 二级指标 · `perspective`（切分型，单位 ACC）

按叙事视角切分（meta.perspective）。

- `first_person`：以第一人称叙述
- `third_person`：以第三人称叙述

### 二级指标 · `variant`（切分型，单位 ACC）

按难度变体切分（meta.variant）。

- `base`：全量基础题
- `hardest`：dim-3 加难版
- `varA`：其他改写变体 A
- `varB`：其他改写变体 B

### 二级指标 · `length`（切分型，单位 ACC）

按情景文本长短切分（meta.length_mode）。

- `long`：长情景文本
- `short`：短情景文本

### 二级指标 · `q4_score`（汇总型，单位 0-10 均分）

Q4 rubric 平均分（0–10 分，非 0–1 准确率）。

**计算方式**：仅取 open 且有 judge_score 的样本，按 0–10 rubric 求均分。overall=全部 Q4 均分，其余 split 按 meta.dim（三级维度）分组。

- `overall`：全部 Q4 的 rubric 平均分

## 人工审核与 qualified 镜像

| 指标 | 定义 |
|---|---|
| review_pass_count | 人工审核合格（meta.review_pass=True）样本数。 |
| review_fail_count | 审核不合格样本数。 |
| review_pass_rate | 合格率 = review_pass_count / total。 |

> `qualified` 是一份镜像：仅在审核合格样本上**重算同一套**一级/二级/三级/四级指标，结构与上文完全一致（accuracy + dimensions 树）。v5.3 默认全部合格时，qualified 与全量一致。
