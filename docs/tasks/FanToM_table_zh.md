# FanToM — 指标定义（全中文版）

- 指标层级与 `metrics.json` 的 `avg_metrics.dimensions` 树一一对应：
  **一级**=总体 accuracy；**二级**=各维度；**三级 / 四级**=维度内嵌套子维度。
- `切分型`：把数据集切成多个 split，各 split 一条准确率；`汇总型`：任务特有口径（如配对联合、宏平均、set 级 ALL），单值无法从边际准确率反推。
- 本页只列指标定义与逐值释义，不含具体数值。
- 本页为全中文版：指标名 / 字段值一律中文（英文标识符括注在后）。英文技术版见 [FanToM_table.md](FanToM_table.md)。

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
|  | `question_type` | — | — |
|  | `order` | — | — |
|  | `set_all` | — | — |

## 各维度定义

### 二级指标 · 题型（type）（切分型，单位 ACC）

题型维度（所有数据集固定带）。按 prompt_type 切分，每个 split 是该题型的准确率。

- 单选（`mcq_single`）：仅一个正确项 + 干扰项的选择题
- 多选（`mcq_multi`）：有 ≥2 个正确项的选择题
- 捆绑判分（`mcq_grouped`）：一条 prompt 内含多个子问题，全部答对才计为对（如 EmoBench EU 的情绪+原因）
- 开放题（`open`）：无干扰项、自由作答，按 f1 / rubric 判分后二值化

### 二级指标 · 题型（question_type）（切分型，单位 ACC）

按题型切分（meta.question_type）。

- 信念题（`beliefQAs`）：角色对事实的信念（标准化为多选形式）
- 可答性-二元（`answerabilityQAs_binary`）：该问题在给定信息下谁能回答（是非）
- 可答性-列举（`answerabilityQA_list`）：列举能回答该问题的角色
- 信息可达-二元（`infoAccessibilityQAs_binary`）：谁获知了该信息（是非）
- 信息可达-列举（`infoAccessibilityQA_list`）：列举获知该信息的角色
- 事实控制（`factQA`）：事实核对控制项（非 ToM 题）

### 二级指标 · ToM 阶数（order）（切分型，单位 ACC）

按 ToM 阶数切分（meta.order）。

- 事实/前置（`0`）：事实或一阶前置
- 一阶信念（`1`）：一阶信念推理
- 二阶信念（`2`）：二阶信念推理

### 二级指标 · set 级 ALL（set_all）（汇总型，单位 ACC）

FANToM 官方头条指标：同一 info-set 内指定 ToM 题型全部答对才算该 set 通过。

**计算方式**：按 snippet(info-set) 用 group_all_correct 分组，要求指定题型全部出现且全对；通过 set 数 / 总 set 数。对应官方 All(MC belief)。

- 总体（`overall`）：全部 ToM 题型都答对才通过
- 可答性子集（`answerability`）：仅要求 answerability 两题型全对
- 信息可达子集（`infoaccess`）：仅要求 infoAccessibility 两题型全对
