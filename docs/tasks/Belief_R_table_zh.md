# Belief_R — 指标定义（全中文版）

- 指标层级与 `metrics.json` 的 `avg_metrics.dimensions` 树一一对应：
  **一级**=总体 accuracy；**二级**=各维度；**三级 / 四级**=维度内嵌套子维度。
- `切分型`：把数据集切成多个 split，各 split 一条准确率；`汇总型`：任务特有口径（如配对联合、宏平均、set 级 ALL），单值无法从边际准确率反推。
- 本页只列指标定义与逐值释义，不含具体数值。
- 本页为全中文版：指标名 / 字段值一律中文（英文标识符括注在后）。英文技术版见 [Belief_R_table.md](Belief_R_table.md)。

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
|  | `step` | — | — |
|  | `modus` | — | — |
|  | `types_of_relation` | — | — |
|  | `belief_reasoning` | — | — |

## 各维度定义

### 二级指标 · 题型（type）（切分型，单位 ACC）

题型维度（所有数据集固定带）。按 prompt_type 切分，每个 split 是该题型的准确率。

- 单选（`mcq_single`）：仅一个正确项 + 干扰项的选择题
- 多选（`mcq_multi`）：有 ≥2 个正确项的选择题
- 捆绑判分（`mcq_grouped`）：一条 prompt 内含多个子问题，全部答对才计为对（如 EmoBench EU 的情绪+原因）
- 开放题（`open`）：无干扰项、自由作答，按 f1 / rubric 判分后二值化

### 二级指标 · 推理步骤（step）（切分型，单位 ACC）

按信念推理步骤切分（meta.step 归一）。

- 信念修正（`belief_update`）：需根据新信息更新既有信念（meta.step=time_t1）
- 信念匹配（`belief_matching`）：直接匹配已陈述的信念、无需更新（meta.step=time_t）

### 二级指标 · 推理式（modus）（切分型，单位 ACC）

按逻辑推理式切分（meta.modus）。

- 肯定前件（`ponens`）：由「若 A 则 B」与 A，推出 B
- 否定后件（`tollens`）：由「若 A 则 B」与 ¬B，推出 ¬A

### 二级指标 · 关系类型（types_of_relation）（切分型，单位 ACC）

按条件规则的关系类型切分（meta.types_of_relation）。

- 事件→事件（`If-Event-Then-Event`）：前件后件都是事件
- 事件→心理状态（`If-Event-Then-MentalState`）：后件是心理状态（信念/情绪等）

### 二级指标 · 信念推理（belief_reasoning）（汇总型，单位 ACC）

Belief-R 官方头条指标，把信念修正/匹配两子集汇总。

**计算方式**：BREU = (BU-Acc + BM-Acc) / 2，是两子集准确率的**宏平均**，不是全样本 ACC；n 为两子集样本合计。

- 信念推理综合（`BREU`）：BU-Acc 与 BM-Acc 的宏平均（官方 BREU）
- 信念修正准确率（`BU-Acc`）：belief_update 子集准确率
- 信念匹配准确率（`BM-Acc`）：belief_matching 子集准确率
