# ExploreToM — 指标定义

- 指标层级与 `metrics.json` 的 `avg_metrics.dimensions` 树一一对应：
  **一级**=总体 accuracy；**二级**=各维度；**三级 / 四级**=维度内嵌套子维度。
- `切分型`：把数据集切成多个 split，各 split 一条准确率；`汇总型`：任务特有口径（如配对联合、宏平均、set 级 ALL），单值无法从边际准确率反推。
- 本页只列指标定义与逐值释义，不含具体数值。
- 全中文版见 [ExploreToM_table_zh.md](ExploreToM_table_zh.md)。

> 说明：旧版声明的 `difficulty` / `task_type` / `order` 在标准化 meta 中并不存在（恒为 unknown），已替换为真实字段 `answer_type` / `nth_order` / `story_type`。

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
|  | `dimension` | — | — |
|  | `answer_type` | — | — |
|  | `nth_order` | — | — |
|  | `story_type` | — | — |

## 各维度定义

### 二级指标 · `type`（切分型，单位 ACC）

题型维度（所有数据集固定带）。按 prompt_type 切分，每个 split 是该题型的准确率。

- `mcq_single`：仅一个正确项 + 干扰项的选择题
- `mcq_multi`：有 ≥2 个正确项的选择题
- `mcq_grouped`：一条 prompt 内含多个子问题，全部答对才计为对（如 EmoBench EU 的情绪+原因）
- `open`：无干扰项、自由作答，按 f1 / rubric 判分后二值化

### 二级指标 · `dimension`（切分型，单位 ACC）

按考察维度切分（meta.dimension）。

- `belief`：一般信念追踪题
- `false_belief`：错误信念题（核心 ToM）

### 二级指标 · `answer_type`（切分型，单位 ACC）

按答案形式切分（meta.answer_type）。

- `binary_knows`：判断「某角色是否知道」的是非题
- `binary_yesno`：一般是非题
- `location`：回答物体所在位置（开放作答）

### 二级指标 · `nth_order`（切分型，单位 ACC）

按信念阶数切分（meta.nth_order）。

- `1`：对他人信念的推理
- `2`：对「他人关于他人信念」的推理
- `-1`：非信念阶（事实/记忆类）

### 二级指标 · `story_type`（切分型，单位 ACC）

按故事生成模板切分（meta.story_type）。

由 ToMi/FANToM 等模板及其变体组合而成（tomi*/fantom-public/fantom-private/all* 等），后缀 +asymmetric 表示信息不对称加强。共 18 种。
