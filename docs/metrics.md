# 指标体系（Metrics）

本文件讲清楚 ToMEval **如何把模型的逐题判分聚合成分层指标**：一级总体准确率、二级维度、三级 / 四级嵌套子维度，以及若干任务特有的「汇总型」口径。读完应能看懂任意数据集 `results/<DS>/<model>/exp_*/metrics.json` 里的每一个数字。

- 想了解各数据集考察什么、从哪改造 → 看 [datasets.md](datasets.md)
- 想了解四个评测协议（direct / direct_think / cot / del_tom）→ 看 [protocols.md](protocols.md)
- 想了解出图 → 看 [visualization.md](visualization.md)

---

## 1. 一句话概览

所有数据集跑同一条流水线，judge 阶段对每条样本产出 `{is_correct, ...}`，再由各数据集的 `tasks/<DS>/metrics.py` 聚合。**聚合采用统一的递归模型：一切皆「维度」，维度可任意嵌套。** 因此 19 个数据集的 `metrics.json` 结构完全一致，只是维度种类不同。

```
avg_metrics
├── accuracy / correct / total          ← 一级指标（总体）
├── extraction_failed / _rate           ← 一级附属（格式遵循诊断）
└── dimensions                          ← 二级及以下
    └── <维度名>: { <split>: { acc, n,
                                dimensions? } }   ← split 内可再挂三级、四级…
```

---

## 2. 一级指标

| 指标 | 含义 |
|------|------|
| `accuracy` | 总体准确率 = `correct / total`。open 题按各数据集判分模式（f1 / rubric）二值化后一并计入。 |
| `correct` | 答对样本数。 |
| `total` | 参与评测的样本总数。 |
| `extraction_failed` | 答案抽取失败数。MCQ 严格模式下模型未输出 `\boxed{}` 即记为抽取失败。 |
| `extraction_failed_rate` | `extraction_failed / total`，衡量模型对输出格式的遵循程度，与对错无关。 |

> 一级指标在所有数据集都存在，由 `base_metric_payload()` 统一产出。

---

## 3. 维度（二级及以下）

### 3.1 统一模型：一切皆维度

一个**维度**（dimension）= 一种切分口径，把样本分成若干 **split**，每个 split 一条记录 `{acc, n}`：

```json
"dimensions": {
  "order": {
    "0": {"acc": 0.75, "n": 8},
    "1": {"acc": 0.75, "n": 4},
    "2": {"acc": 0.27, "n": 11}
  }
}
```

- `acc`：该 split 的准确率（个别维度放的是其它标量，见 §3.4）。
- `n`：该 split 的样本数。

每个数据集都**固定带一个 `type` 维度**（按题型 `prompt_type` 切分：`mcq_single` / `mcq_multi` / `mcq_grouped` / `open`），其余维度由各 `metrics.py` 声明。

### 3.2 切分型 vs 汇总型

| 类型 | 含义 | 例子 |
|------|------|------|
| **切分型** | 把整个数据集切成多个 split，各 split 一条准确率。可从逐样本反推。 | `type`、`order`、`dimension`、`ability`… |
| **汇总型** | 任务特有口径，**无法从边际准确率反推**，由任务算好后用 `add_dimension()` 直接挂上。通常只有一个 `overall` split。 | Belief-R `belief_reasoning`、BigToM `tb_and_fb`、FanToM `set_all`、TactfulToM `joint_comp_just`、FictionalQA `grading` / `macro_split_acc` |

> 为什么需要汇总型？例如 BigToM 的 `tb_and_fb`：要求**同一故事**的 true-belief 与 false-belief 两问都对才算通过。这是「配对联合」口径，单看 TB、FB 两个边际准确率算不出来，必须在聚合时按故事配对统计。

#### 合取判分的两种实现：生成时合并 vs 聚合时分组

「多个子问题全部答对才算对」这种**合取（AND）判分**在框架里有两套实现，**容易混淆但不可互换**。判据只有一条：**模型应不应该同时看到这些子问题？**

| | `mcq_grouped`（一种题型） | 汇总型维度（`group_all_correct`） |
|---|---|---|
| 合并发生在 | **生成时**：多个子问题拼进**同一个 prompt**，模型一次性作答 | **聚合时**：每个子问题是**独立样本、独立 prompt、独立生成** |
| 由谁合并 | `prepare_samples` 预打包进 `meta.sub_questions` | judge 阶段按 `set_id` / 故事 / 对话分组 |
| 如何判分 | `rule_judge_grouped`：抽末尾 N 个 `\boxed{}`，全对才 `is_correct` | 各样本各自判分，再 `group_all_correct` 要求组内成员全对 |
| 代表 | EmoBench EU（情绪+原因捆绑问） | FanToM `set_all`、BigToM `tb_and_fb`、TactfulToM `joint_comp_just` |

- **该一起问 → `mcq_grouped`**。EmoBench EU 的「原因」依赖先认出「情绪」，原 benchmark 本就捆绑呈现，拼成一个 prompt 是忠实复刻。
- **必须分开问 → 独立样本 + 汇总型**。这几个数据集换成 `mcq_grouped` 会**改变指标本身**：
  - FanToM `set_all` 官方定义就是 `groupby(set_id).all().mean()`，刻意让每题独立问（模型看不到同 set 其它题），考的正是「跨独立提问的一致性」；合进一个 prompt 会让模型互相参照，分数虚高、指标变味。
  - BigToM `tb_and_fb` 的 TB / FB 是**两个不同的故事变体**，物理上无法塞进同一 prompt，只能按同源场景配对。
  - TactfulToM `joint_comp_just` 的 Comp / Just 是 Happé 双题协议，本就独立问、再要求都对。

> 反向的例子：EmoBench **两者都用**。它用 `mcq_grouped` 把 EU 的情绪+原因合判（往上合），又有汇总型 `eu_subquestion` 把 grouped 样本的 `sub_results` 拆回情绪 / 原因各自准确率（往下拆）。后者算汇总型，是因为这个拆解值藏在样本内部，从顶层 `is_correct` 反推不出来。

### 3.3 三级 / 四级：维度嵌套

任意 split 内部还能再切，子维度挂在该 split 的 `dimensions` 下，**层数不限**。目前最深到四级：

- **SoMBench**（四级）：`dim1`（能力大类）→ `dim2`（二级标签如 1.1）→ `dim3`（三级标签如 1.1.2）
- **EmoBench**（三级）：`coarse_category`（情绪粗类）→ `finegrained_category`（情绪细类）

> 注：ToMato 旧版曾按 `dimension_1→dimension_2→dimension_3` 展开四级，但标准化数据的 `meta.dimension` 实为单槽（仅心智状态一项），后两级恒为占位空桶，已删除，改用 `mental_state` / `order` / `false_belief`。

读 / 写都靠同一套递归（`build_dimension` 生成、`iter_dimension_nodes` 消费），所以「到三级就截断」的情况不存在。**一个二级维度也可以同时挂多个不同的三级维度**（多个 `sub_specs`），它们都是该二级的并列子维度。

### 3.4 特殊量纲：非准确率的维度

个别维度的 `acc` 字段放的不是 0–1 准确率，而是其它标量：

- **SoMBench `q4_score`**：Q4 开放分析题的 rubric 平均分，量程 **0–`max_score`**（默认 10）。`overall` = 全部 Q4 均分，其余 split 按三级维度 `meta.dim` 分组。
- **FictionalQA `grading.blind`**：无上下文盲评的均分。

> **0–10 与 0–1 的约定**：rubric **只在评测/judge 阶段按 10 分制计算**（满分 `max_score`=10，`open_threshold`=7 判过），`judge_score`、`q4_score` 都按原始 0–10 分存进 `metrics.json`。下游凡需要 0–1 口径的场合（与其它准确率同图、跨数据集汇总、归一对比），**统一除以 `max_score` 归一**即可：`q4_norm = q4_score / 10`。换言之，存储保留满分制以免精度损失与语义丢失，归一是消费侧的一步显式换算。
>
> ⚠️ 注意：当前 `plots.py` 的 `iter_dimension_nodes` 出图路径会把 `q4_score` 当普通准确率维度、走写死 `ylim(0, 1)` 的柱状图，0–10 分的柱子会顶破 y 轴。出图前对这类维度先 `/max_score` 归一（或单独走 `plot_score_group` 的自适应 y 轴）即可，**表格层 `flatten_payload_dimensions` 按原值如实摊出、不受影响**。

---

## 4. open 题判分（影响一级 accuracy）

MCQ 走 `\boxed{}` 规则提取判分；open 题没有干扰项，按数据集 `config.yaml` 的 `open_judge` 字段选判分模式：

| 模式 | 需要 judge model | 判定方式 | 阈值 | 用在 |
|------|:---:|------|------|------|
| `f1` | 否 | token / 字符级 F1 对比参考答案，过阈值算对 | `f1_threshold`，默认 **0.5** | ExploreToM · FictionalQA · SocialBench · ToMChallenges |
| `llm_simple` | 是（judge1） | 二元 LLM judge，参照式输出 is_correct | — | （当前无） |
| `rubric` | 是（judge1[+2]） | 数据集 rubric prompt 打总分，多 judge 取平均，过阈值算对 | `open_threshold`，默认 **7.0** | SoMBench（满分 10） |

判分模式是**数据集答案格式的内在属性**（短答案→f1、社会认知长答案→rubric），所以配置放在数据集侧而非全局。

> open 判分的结果（对 / 错）会并入一级 `accuracy`。判分失败（如 judge 调用报错）的样本记为错误并带 `error_reason`，不会静默丢弃。

---

## 5. SoMBench 专属：qualified 镜像

SoMBench 在常规指标之外，额外产出一份 **`qualified` 镜像**：仅在「人工审核合格」（`meta.review_pass=True`）的样本上**重算同一套**一级 / 二级 / 三级 / 四级指标，结构与全量完全一致。

| 字段 | 含义 |
|------|------|
| `review_pass_count` / `review_fail_count` | 审核合格 / 不合格样本数。 |
| `review_pass_rate` | 合格率 = `review_pass_count / total`。 |
| `qualified` | 合格样本上的完整指标树（`accuracy` + `dimensions` + …）。 |

> v5.3 的 provenance 标明 FAIL 项已就地修复、裁判结论为最终权威，故默认全部样本合格，`qualified` 与全量一致。缺 `review_pass` 字段的样本默认视为合格（保证其它数据集不受影响）。

---

## 6. 各数据集的维度一览

下表列出每个数据集除固定 `type` 外声明的维度。**缩进**表示嵌套（`→` 子维度）；标 *(汇总)* 的是汇总型，其余为切分型。

| 数据集 | 维度（切分依据） |
|--------|------------------|
| **Belief_R** | `step`（belief_update / belief_matching）；`modus`（ponens / tollens）；`types_of_relation`（事件→事件 / 事件→心理状态）；`belief_reasoning` *(汇总：BREU=BU/BM 均值，BU-Acc=信念修正，BM-Acc=信念匹配)* |
| **BigToM** | `condition`（forward/backward belief 等因果模板条件）；`belief_type`（true/false belief·control）；`tb_and_fb` *(汇总：同一故事 TB∧FB 两问全对)* |
| **EmoBench** | `subset`（EU/EA）；`language`（en/zh）；`question_subtype`；`coarse_category` → `finegrained_category`（情绪粗类→细类）；`dimension`（细粒度能力，可多值）；`eu_subquestion` *(汇总：EU 子问题 emotion/cause 各自准确率)* |
| **ExploreToM** | `dimension`（belief / false_belief）；`answer_type`（binary_knows / binary_yesno / location）；`nth_order`（-1 / 1 / 2）；`story_type`（tomi/fantom 模板 18 种） |
| **FanToM** | `question_type`（belief/answerability/infoaccess 各题型）；`order`（0/1/2 ToM 阶数）；`set_all` *(汇总：同一 info-set 内 ToM 题型全对，含 overall/answerability/infoaccess)* |
| **FictionalQA** | `style`（虚构文体）；`grading` *(汇总：informed vs blind 盲评，差值即 gap)*；`macro_split_acc` *(汇总：event/document/style 三种口径的宏平均)* |
| **HellaSwag** | `split_type`（indomain / zeroshot） |
| **HiToM** | `order`（心智推理阶数 0–4，核心难度轴） |
| **PUB** | `option_count`（选项个数 2–5）。⚠️ 转换后 meta 稀薄，原 source/difficulty/ethics_category/task_type/14 子任务信息已丢失 |
| **SimpleToM** | `dimension`（information_access / behavior_prediction / social_judgment）；`qa_type`（mental_state / behavior / judgment）；`scenario_name`（10 类日常情景） |
| **SocialBench** | `category`（官方 <层级>-<能力>-<子任务> 类别码）；`dimension`（conversation_memory / self_awareness / …，可多值）；`lang`（en/zh）；`num_choices`（0=开放题） |
| **SocialIQA** | `dimension`（ATOMIC 九维：xIntent/xNeed/xAttr/…/oWant） |
| **TactfulToM** | `category`（belief/answerability/…）；`question_type`（完整路径如 tom:belief:accessible:truth）；`lie_type`（altruistic / pareto 白谎）；`tom_type`（一阶/二阶各角色对）；`joint_comp_just` *(汇总：同一对话 Comp∧Just 都对，Happé 双题判定)* |
| **ToMBench** | `task`（对应官方 Task-oriented：False-Belief-Task 等）；`ability`（Belief/Desire/Emotion/Intention/Knowledge/Non-Literal Communication）；`lang`（en/zh） |
| **ToMChallenges** | `question_type`（1stA/1stB/2ndA/2ndB/assumption/memory/reality）；`task_format`（mc / qa）；`test_type`（sally-anne / smarties） |
| **ToMQA** | `dimension`（belief / memory / reality / search）；`task`（fb / tb / sofb） |
| **ToMato** | `mental_state`（belief/desire/emotion/intention/knowledge）；`order`（1 / 2）；`false_belief`（True / False） |
| **ToMi** | `story_type`（true/false/second_order false belief）；`question_type`（一/二阶 × 是否需 ToM、memory、reality） |
| **SoMBench** | `dim1` → `dim2` → `dim3`（能力维度三级体系）；`qtype`（Q1–Q4）；`perspective`（first/third person）；`variant`（base/hardest/varA/varB）；`length`（long/short）；`q4_score` *(汇总：Q4 rubric 0–10 均分)*；另含 `qualified` 镜像（见 §5） |

> 上表的「维度名」即 `metrics.json` 里 `dimensions` 的键，与各 `tasks/<DS>/metrics.py` 声明一致。每个数据集的完整逐值释义（含中文副本）见 [tasks/](tasks/README.md) 下的 `<DS>_table.md` / `<DS>_table_zh.md`。各 split 的具体取值随实验而变，不在本文件固化；要看某次实验的实际数字，直接读对应的 `metrics.json`，或用可视化（[visualization.md](visualization.md)）出图。

---

## 7. 代码索引

| 关注点 | 位置 |
|--------|------|
| 分层聚合核心（`hierarchical_metrics` / `build_dimension` / `add_dimension` / `make_split`） | `src/evaluation/task_metrics.py` |
| 各数据集维度声明（`compute_metrics`） | `tasks/<DS>/metrics.py` |
| open 判分（f1 / llm_simple / rubric） | `src/evaluation/open_judge.py` |
| judge 阶段编排 | `src/evaluation/pipeline.py`（`run_metric_stage`） |
| 消费 dimensions 树出图 | `src/visualization/plots.py`（`iter_dimension_nodes`） |
| 消费 dimensions 树出表 | `report/generate_dataset_tables.py`（`flatten_payload_dimensions`） |
