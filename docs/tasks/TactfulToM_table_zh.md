# TactfulToM — 指标定义（全中文版）

- 指标层级与 `metrics.json` 的 `avg_metrics.dimensions` 树一一对应：
  **一级**=总体 accuracy；**二级**=各维度；**三级 / 四级**=维度内嵌套子维度。
- `切分型`：把数据集切成多个 split，各 split 一条准确率；`汇总型`：任务特有口径（如配对联合、宏平均、set 级 ALL），单值无法从边际准确率反推。
- 本页只列指标定义与逐值释义，不含具体数值。
- 本页为全中文版：指标名 / 字段值一律中文（英文标识符括注在后）。英文技术版见 [TactfulToM_table.md](TactfulToM_table.md)。

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
|  | `question_type` | — | — |
|  | `lie_type` | — | — |
|  | `tom_type` | — | — |
|  | `joint_comp_just` | — | — |

## 各维度定义

### 二级指标 · 题型（type）（切分型，单位 ACC）

题型维度（所有数据集固定带）。按 prompt_type 切分，每个 split 是该题型的准确率。

- 单选（`mcq_single`）：仅一个正确项 + 干扰项的选择题
- 多选（`mcq_multi`）：有 ≥2 个正确项的选择题
- 捆绑判分（`mcq_grouped`）：一条 prompt 内含多个子问题，全部答对才计为对（如 EmoBench EU 的情绪+原因）
- 开放题（`open`）：无干扰项、自由作答，按 f1 / rubric 判分后二值化

### 二级指标 · 问题大类（category）（切分型，单位 ACC）

按问题大类切分（meta.category）。

- 信念（`belief`）：对角色信念的判断
- 可答性（`answerability`）：问题在给定信息下是否可答
- 信息可达性（`info_accessibility`）：角色是否能获知某信息
- 善意谎言点（`lieability`）：识别谁会/该说善意谎言
- 谎言可识别性（`liedetectability`）：谎言是否可被识破
- 辩护（`justification`）：对善意谎言动机的解释判断（与 comprehension 配对做联合）
- 对话理解（`comprehension`）：对话事实理解题（与 justification 配对）
- 事实控制（`fact`）：事实核对控制项

### 二级指标 · 细粒度题型（question_type）（切分型，单位 ACC）

按细粒度题型完整路径切分（meta.question_type）。

路径格式 <层级>:<能力>:<形式/可达性>:<truth 真话 / real_reason 真因 / reason 理由>，如 tom:belief:accessible:truth。共 23 种。

### 二级指标 · 白谎类型（lie_type）（切分型，单位 ACC）

按白谎类型切分（meta.lie_type）。

- 利他型善意谎言（`altruistic_white_lies`）：纯为他人利益的善意谎言
- 帕累托型善意谎言（`pareto_white_lies`）：利他且不损己的双赢善意谎言

### 二级指标 · ToM 阶数×角色（tom_type）（切分型，单位 ACC）

按 ToM 阶数与角色组合切分（meta.tom_type）。

形如 first-order:A / second-order:AB，字母为角色代号；空值为非 ToM 题。共 13 种。

### 二级指标 · 理解∧辩护联合（joint_comp_just）（汇总型，单位 ACC）

Comp∧Just 联合指标：同一对话的对话理解与辩护两题都答对，才算真正理解了善意谎言（Happé 双题判定）。

**计算方式**：按 set_id 用 group_all_correct 分组，要求同时含 comprehension、justification；两者皆对的对话数 / 总对话数。无法由两个边际 ACC 反推。

- 总体（`overall`）：全部对话上的联合通过率
