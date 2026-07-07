# FictionalQA — 指标定义（全中文版）

- 指标层级与 `metrics.json` 的 `avg_metrics.dimensions` 树一一对应：
  **一级**=总体 accuracy；**二级**=各维度；**三级 / 四级**=维度内嵌套子维度。
- `切分型`：把数据集切成多个 split，各 split 一条准确率；`汇总型`：任务特有口径（如配对联合、宏平均、set 级 ALL），单值无法从边际准确率反推。
- 本页只列指标定义与逐值释义，不含具体数值。
- 本页为全中文版：指标名 / 字段值一律中文（英文标识符括注在后）。英文技术版见 [FictionalQA_table.md](FictionalQA_table.md)。

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
|  | `style` | — | — |
|  | `grading` | — | — |
|  | `macro_split_acc` | — | — |

## 各维度定义

### 二级指标 · 题型（type）（切分型，单位 ACC）

题型维度（所有数据集固定带）。按 prompt_type 切分，每个 split 是该题型的准确率。

- 单选（`mcq_single`）：仅一个正确项 + 干扰项的选择题
- 多选（`mcq_multi`）：有 ≥2 个正确项的选择题
- 捆绑判分（`mcq_grouped`）：一条 prompt 内含多个子问题，全部答对才计为对（如 EmoBench EU 的情绪+原因）
- 开放题（`open`）：无干扰项、自由作答，按 f1 / rubric 判分后二值化

### 二级指标 · 文体（style）（切分型，单位 ACC）

按虚构文体切分（meta.style）。

- 新闻（`news`）：新闻稿体
- 企业公文（`corporate`）：企业/公文体
- 百科（`encyclopedia`）：百科词条体
- 博客（`blog`）：博客文体
- 社媒（`social`）：社交媒体文体

### 二级指标 · 有上下文 vs 盲评（grading）（汇总型，单位 ACC）

informed-vs-blind 对照，衡量「给了虚构上下文」相对「盲评」的增益。

**计算方式**：informed = 全样本 ACC；blind = 对每题的 meta.blind_grade_avg 求平均（无上下文盲评均分）；两者之差即官方关注的 gap。

- 有上下文（`informed`）：模型在给定虚构上下文下的准确率
- 盲评（`blind`）：无上下文时的盲评均分

### 二级指标 · 宏平均准确率（macro_split_acc）（汇总型，单位 ACC）

三种分组口径下的宏平均，消除大组主导。

**计算方式**：每种口径先按组算组内 ACC，再对各组**等权平均**（n=组数），而非全样本 ACC。

- 按事件（`event`）：先按 event 分组算 ACC 再跨组平均
- 按文档（`document`）：先按 document 分组算 ACC 再跨组平均
- 按文体（`style`）：先按 style 分组算 ACC 再跨组平均
