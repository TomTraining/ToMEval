# EmoBench — 指标定义（全中文版）

- 指标层级与 `metrics.json` 的 `avg_metrics.dimensions` 树一一对应：
  **一级**=总体 accuracy；**二级**=各维度；**三级 / 四级**=维度内嵌套子维度。
- `切分型`：把数据集切成多个 split，各 split 一条准确率；`汇总型`：任务特有口径（如配对联合、宏平均、set 级 ALL），单值无法从边际准确率反推。
- 本页只列指标定义与逐值释义，不含具体数值。
- 本页为全中文版：指标名 / 字段值一律中文（英文标识符括注在后）。英文技术版见 [EmoBench_table.md](EmoBench_table.md)。

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
|  | `subset` | — | — |
|  | `language` | — | — |
|  | `question_subtype` | — | — |
|  | `coarse_category` | `finegrained_category` | — |
|  | `dimension` | — | — |
|  | `eu_subquestion` | — | — |

## 各维度定义

### 二级指标 · 题型（type）（切分型，单位 ACC）

题型维度（所有数据集固定带）。按 prompt_type 切分，每个 split 是该题型的准确率。

- 单选（`mcq_single`）：仅一个正确项 + 干扰项的选择题
- 多选（`mcq_multi`）：有 ≥2 个正确项的选择题
- 捆绑判分（`mcq_grouped`）：一条 prompt 内含多个子问题，全部答对才计为对（如 EmoBench EU 的情绪+原因）
- 开放题（`open`）：无干扰项、自由作答，按 f1 / rubric 判分后二值化

### 二级指标 · 子集（subset）（切分型，单位 ACC）

按官方子集切分（meta.subset）。

- 情绪理解(EU)（`emotional_understanding`）：判断当事人的情绪及其成因
- 情绪应用(EA)（`emotional_application`）：在情境中选择合适的行动/回应

### 二级指标 · 语种（language）（切分型，单位 ACC）

按语种切分（meta.language）。

- 英文（`en`）：英文题
- 中文（`zh`）：中文题

### 二级指标 · 问题子类型（question_subtype）（切分型，单位 ACC）

按问题子类型切分（meta.question_subtype）。

- 情绪（`emotion`）：判断当事人的情绪
- 原因（`cause`）：判断情绪的成因
- 行动（`Action`）：EA：应采取的行动
- 回应（`Response`）：EA：应作出的回应

### 二级指标 · 情绪粗类（coarse_category）（切分型，单位 ACC）

情绪理解(EU)的粗类（meta.coarse_category），其下嵌套 finegrained_category（EA 子集无此标签）。

- 复杂情绪（`complex_emotions`）：混合/转变类复杂情绪
- 情绪线索（`emotional_cues`）：从视觉/言语线索推断情绪
- 个人信念与经历（`personal_beliefs_and_experiences`）：受个人信念、文化、经历影响的情绪
- 观点采择（`perspective_taking`）：站在他人视角推断情绪
- 未分类（`unknown`）：情绪应用(EA)子集样本无粗类标签，归为 unknown

### 三级指标 · 情绪细类（coarse_category → finegrained_category）（切分型，单位 ACC）

情绪理解细类（meta.finegrained_category），挂在各 coarse_category split 下。

- 混合情绪（`mixture_of_emotions`）：同时存在多种情绪
- 情绪转变（`emotion_transition`）：情绪随情节变化
- 意外结局（`unexpected_outcome`）：结局出人意料引发的情绪
- 视觉线索（`visual_cues`）：由表情/动作等视觉线索判断
- 言语线索（`vocal_cues`）：由语气/措辞等线索判断
- 文化价值（`cultural_value`）：受文化价值观影响的情绪
- 情感价值（`sentimental_value`）：受物品/回忆的情感价值影响
- 人物设定（`persona`）：依据人物设定推断情绪
- 错误信念（`false_belief`）：基于错误信念的情绪
- 失礼（`faux_pas`）：失礼情境下的情绪
- 奇异故事（`strange_story`）：Happé 奇异故事式情境
- 未分类（`unknown`）：情绪应用(EA)子集样本无细类标签，归为 unknown

### 二级指标 · 能力标签（dimension）（切分型，单位 ACC）

细粒度能力标签（meta.dimension，可多值），把粗/细类与 EA 子任务平铺成一张总表。

取值为多段组合标签，形如 ['emotional_application','Personal-Others','Action']（EA）或 ['emotional_understanding', 粗类, 细类]（EU）。

### 二级指标 · EU 子问题（eu_subquestion）（汇总型，单位 ACC）

EU 子问题诊断：从 mcq_grouped 记录的 sub_results 拆出情绪/原因两类子问题。

**计算方式**：遍历所有 mcq_grouped 记录的 sub_results，按 subtype(emotion/cause) 分别累计 correct/total，得两条子问题准确率。

- 情绪判断（`emotion`）：EU 情绪子问题准确率
- 原因判断（`cause`）：EU 原因子问题准确率
