# EmoBench — 指标定义

- 指标层级与 `metrics.json` 的 `avg_metrics.dimensions` 树一一对应：
  **一级**=总体 accuracy；**二级**=各维度；**三级 / 四级**=维度内嵌套子维度。
- `切分型`：把数据集切成多个 split，各 split 一条准确率；`汇总型`：任务特有口径（如配对联合、宏平均、set 级 ALL），单值无法从边际准确率反推。
- 本页只列指标定义与逐值释义，不含具体数值。
- 全中文版见 [EmoBench_table_zh.md](EmoBench_table_zh.md)。

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
|  | `subset` | — | — |
|  | `language` | — | — |
|  | `question_subtype` | — | — |
|  | `coarse_category` | `finegrained_category` | — |
|  | `dimension` | — | — |
|  | `eu_subquestion` | — | — |

## 各维度定义

### 二级指标 · `type`（切分型，单位 ACC）

题型维度（所有数据集固定带）。按 prompt_type 切分，每个 split 是该题型的准确率。

- `mcq_single`：仅一个正确项 + 干扰项的选择题
- `mcq_multi`：有 ≥2 个正确项的选择题
- `mcq_grouped`：一条 prompt 内含多个子问题，全部答对才计为对（如 EmoBench EU 的情绪+原因）
- `open`：无干扰项、自由作答，按 f1 / rubric 判分后二值化

### 二级指标 · `subset`（切分型，单位 ACC）

按官方子集切分（meta.subset）。

- `emotional_understanding`：判断当事人的情绪及其成因
- `emotional_application`：在情境中选择合适的行动/回应

### 二级指标 · `language`（切分型，单位 ACC）

按语种切分（meta.language）。

- `en`：英文题
- `zh`：中文题

### 二级指标 · `question_subtype`（切分型，单位 ACC）

按问题子类型切分（meta.question_subtype）。

- `emotion`：判断当事人的情绪
- `cause`：判断情绪的成因
- `Action`：EA：应采取的行动
- `Response`：EA：应作出的回应

### 二级指标 · `coarse_category`（切分型，单位 ACC）

情绪理解(EU)的粗类（meta.coarse_category），其下嵌套 finegrained_category（EA 子集无此标签）。

- `complex_emotions`：混合/转变类复杂情绪
- `emotional_cues`：从视觉/言语线索推断情绪
- `personal_beliefs_and_experiences`：受个人信念、文化、经历影响的情绪
- `perspective_taking`：站在他人视角推断情绪
- `unknown`：情绪应用(EA)子集样本无粗类标签，归为 unknown

### 三级指标 · `coarse_category → finegrained_category`（切分型，单位 ACC）

情绪理解细类（meta.finegrained_category），挂在各 coarse_category split 下。

- `mixture_of_emotions`：同时存在多种情绪
- `emotion_transition`：情绪随情节变化
- `unexpected_outcome`：结局出人意料引发的情绪
- `visual_cues`：由表情/动作等视觉线索判断
- `vocal_cues`：由语气/措辞等线索判断
- `cultural_value`：受文化价值观影响的情绪
- `sentimental_value`：受物品/回忆的情感价值影响
- `persona`：依据人物设定推断情绪
- `false_belief`：基于错误信念的情绪
- `faux_pas`：失礼情境下的情绪
- `strange_story`：Happé 奇异故事式情境
- `unknown`：情绪应用(EA)子集样本无细类标签，归为 unknown

### 二级指标 · `dimension`（切分型，单位 ACC）

细粒度能力标签（meta.dimension，可多值），把粗/细类与 EA 子任务平铺成一张总表。

取值为多段组合标签，形如 ['emotional_application','Personal-Others','Action']（EA）或 ['emotional_understanding', 粗类, 细类]（EU）。

### 二级指标 · `eu_subquestion`（汇总型，单位 ACC）

EU 子问题诊断：从 mcq_grouped 记录的 sub_results 拆出情绪/原因两类子问题。

**计算方式**：遍历所有 mcq_grouped 记录的 sub_results，按 subtype(emotion/cause) 分别累计 correct/total，得两条子问题准确率。

- `emotion`：EU 情绪子问题准确率
- `cause`：EU 原因子问题准确率
