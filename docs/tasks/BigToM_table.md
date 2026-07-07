# BigToM — 指标定义

- 指标层级与 `metrics.json` 的 `avg_metrics.dimensions` 树一一对应：
  **一级**=总体 accuracy；**二级**=各维度；**三级 / 四级**=维度内嵌套子维度。
- `切分型`：把数据集切成多个 split，各 split 一条准确率；`汇总型`：任务特有口径（如配对联合、宏平均、set 级 ALL），单值无法从边际准确率反推。
- 本页只列指标定义与逐值释义，不含具体数值。
- 全中文版见 [BigToM_table_zh.md](BigToM_table_zh.md)。

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
|  | `condition` | — | — |
|  | `belief_type` | — | — |
|  | `tb_and_fb` | — | — |

## 各维度定义

### 二级指标 · `type`（切分型，单位 ACC）

题型维度（所有数据集固定带）。按 prompt_type 切分，每个 split 是该题型的准确率。

- `mcq_single`：仅一个正确项 + 干扰项的选择题
- `mcq_multi`：有 ≥2 个正确项的选择题
- `mcq_grouped`：一条 prompt 内含多个子问题，全部答对才计为对（如 EmoBench EU 的情绪+原因）
- `open`：无干扰项、自由作答，按 f1 / rubric 判分后二值化

### 二级指标 · `condition`（切分型，单位 ACC）

按因果模板生成条件切分（meta.condition_type）。

- `forward_belief`：由感知情境推断角色信念
- `forward_action`：由角色信念预测其行为
- `backward_belief`：由观察到的行为反推角色信念
- `percept_to_belief`：由是否感知到关键事件推断信念

### 二级指标 · `belief_type`（切分型，单位 ACC）

按信念类型切分（meta.dimension 归一化：小写、-/空格转 _）。

- `true_belief`：角色信念与现实一致
- `false_belief`：角色信念与现实不符（核心 ToM 考点）
- `true_control`：去除 ToM 线索的对照项（真）
- `false_control`：去除 ToM 线索的对照项（假）

### 二级指标 · `tb_and_fb`（汇总型，单位 ACC）

配对联合指标：同一故事的真信念题与假信念题都答对才算该 pair 通过。

**计算方式**：按故事分组（去掉 _true_belief/_false_belief 后缀得 pair 键），仅统计同时含 TB、FB 两问的 pair；两问全对的 pair 数 / 总 pair 数。无法由两个边际 ACC 反推。

- `overall`：全部配对故事上的联合通过率
