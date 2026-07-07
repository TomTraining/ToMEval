# SocialBench — 指标定义（全中文版）

- 指标层级与 `metrics.json` 的 `avg_metrics.dimensions` 树一一对应：
  **一级**=总体 accuracy；**二级**=各维度；**三级 / 四级**=维度内嵌套子维度。
- `切分型`：把数据集切成多个 split，各 split 一条准确率；`汇总型`：任务特有口径（如配对联合、宏平均、set 级 ALL），单值无法从边际准确率反推。
- 本页只列指标定义与逐值释义，不含具体数值。
- 本页为全中文版：指标名 / 字段值一律中文（英文标识符括注在后）。英文技术版见 [SocialBench_table.md](SocialBench_table.md)。

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
|  | `category` | — | — |
|  | `dimension` | — | — |
|  | `lang` | — | — |
|  | `num_choices` | — | — |

## 各维度定义

### 二级指标 · 题型（type）（切分型，单位 ACC）

题型维度（所有数据集固定带）。按 prompt_type 切分，每个 split 是该题型的准确率。

- 单选（`mcq_single`）：仅一个正确项 + 干扰项的选择题
- 多选（`mcq_multi`）：有 ≥2 个正确项的选择题
- 捆绑判分（`mcq_grouped`）：一条 prompt 内含多个子问题，全部答对才计为对（如 EmoBench EU 的情绪+原因）
- 开放题（`open`）：无干扰项、自由作答，按 f1 / rubric 判分后二值化

### 二级指标 · 类别（category）（切分型，单位 ACC）

按官方类别码切分（meta.category），格式 <层级>-<能力>-<子任务>。

- 个体-自我认知-角色知识（`Individual-SA-RoleKnowledge`）：角色应知的背景知识
- 个体-自我认知-角色风格（`Individual-SA-RoleStyle`）：角色说话风格一致性
- 个体-情绪感知-对话情绪识别（`Individual-EP-DialogueEmotionDetect`）：识别对话中的情绪
- 个体-情绪感知-幽默讽刺识别（`Individual-EP-HumorSarcasmDetect`）：识别幽默/讽刺
- 个体-情绪感知-情境理解（`Individual-EP-SituationUnderstanding`）：理解情境含义
- 个体-记忆-长程（`Individual-MEM-Long`）：长程对话记忆
- 个体-记忆-短程（`Individual-MEM-Short`）：短程对话记忆
- 群体-社会偏好-正向（`Group-SAP-Positive`）：群体互动中的正向社会偏好
- 群体-社会偏好-中性（`Group-SAP-Neutral`）：中性社会偏好
- 群体-社会偏好-负向（`Group-SAP-Negative`）：负向社会偏好

### 二级指标 · 能力维度（dimension）（切分型，单位 ACC）

按能力维度切分（meta.dimension，可多值）。

- 对话记忆（`conversation_memory`）：记住并利用对话历史
- 自我认知（`self_awareness`）：对自身角色设定的认知
- 社会偏好（`social_preference`）：群体互动中的社会偏好
- 情绪感知（`emotional_perception`）：感知与识别情绪

### 二级指标 · 语种（lang）（切分型，单位 ACC）

按语种切分（meta.lang）。

- 英文（`en`）：英文题
- 中文（`zh`）：中文题

### 二级指标 · 候选数（num_choices）（切分型，单位 ACC）

按候选个数切分（len(options)）。

0 表示开放题（走 f1 判分），其余为该题选项个数。
