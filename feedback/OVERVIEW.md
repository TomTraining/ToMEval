# 数据合成模块概览

> **一句话目标**：从模型在评估集上的错误中学习——自动诊断失败模式，定向合成高质量 ToM 训练数据，再经过去重和质量过滤，输出可直接用于训练的 parquet 文件。

---

## 为什么需要这个模块

直接随机生成训练数据效率低，因为不知道模型在哪里弱。这个模块的核心思路是：

**先评估，再合成**——用 ToMEval 跑出模型的 bad case，分析失败集中在哪些能力维度，然后针对性地生成更多同类型的难题。

```
ToMEval 评估结果（prediction.jsonl）
        │
        ▼
  哪些样本模型答错了？
  多个模型都错的 = 高价值目标
        │
        ▼
  这些错误集中在哪个能力维度？
  错误模式是什么？
        │
        ▼
  针对这些维度，生成新的难题
        │
        ▼
  去重 + 质量过滤
        │
        ▼
  synthetic.parquet → 进入 filter 质量评估
```

---

## 四阶段流水线

### Stage 1：加载 Bad Case

从 ToMEval 的 `results/` 目录读取多个模型的 `prediction.jsonl`，取 bad case **并集**。

关键设计：
- **多模型并集**：任意一个模型答错就纳入，覆盖更广
- **`_hard` 标记**：≥2 个模型都答错的样本标记为 `_hard=True`，是最高价值的合成目标
- **按维度分层**：每个样本附带维度标签（如 `false_belief`、`perspective_taking`），为 Stage 2 分组做准备

输出：`data_output/bad_cases/<dataset>/bad_cases.jsonl`

---

### Stage 2：维度批量诊断

不是逐条分析 bad case，而是**按能力维度分组，批量归纳错误模式**。

每批 K 条同维度的 bad case → 强模型 → `DimensionDiagnosisReport`：

```
DimensionDiagnosisReport {
  dimension:                    "false_belief"
  common_error_patterns:        ["Agent A failed to update belief after unwitnessed event", ...]
  primary_cognitive_operation:  "belief_state_update"
  recommended_synthesis_themes: ["library scenario", "surprise party", ...]
  difficulty_distribution:      {"easy": 0.2, "medium": 0.5, "hard": 0.3}
}
```

这一步的价值：把零散的错误样本提炼成**可复用的合成指令**，而不是简单地改写原题。

输出：`data_output/diagnosis_reports/<dataset>/<split>/dimension_reports.jsonl`

---

### Stage 3：从诊断报告合成新样本

读取 Stage 2 的诊断报告，按报告生成新题目。

关键设计：
- **目标数量驱动**：`target_samples / samples_per_report` 决定需要多少份报告，按维度错误比例分配报告数
- **数据集格式注册表**：每个数据集有独立的 schema（ToMBench 是 4 选 1 内嵌选项，BigToM 是 2 选 1，SocialIQA 是 3 选 1 等），合成时严格遵守
- **难度分布控制**：诊断报告中的 `difficulty_distribution` 指导合成时的难度比例

输出：`data_output/synth_raw/candidates_iter*.jsonl`

---

### Stage 4：LSH 守门员去重

两轮去重，防止数据污染：

| 去重类型 | 做法 | 目的 |
|---|---|---|
| **测试集去重** | MinHash LSH，与测试集比较相似度 | 防止训练数据泄漏到测试集，避免虚假高分 |
| **内部去重** | 合成样本之间互相比较 | 避免重复样本浪费训练资源 |

相似度阈值 0.6，4-gram 字符级 MinHash，128 个哈希函数。

输出：`data_output/synth_clean/<dataset>/synthetic.parquet`

---

## 质量保障：与 filter 的分工

V3 之前，难度验证（Stage 5）在本模块内完成。V3 之后，**难度和质量验证统一交给 `filter` 模块处理**：

```
feedback 输出
  synthetic.parquet
        │
        ▼
  filter 决策树评估
  (pass@k + answerability + shortcut 三探测)
        │
        ▼
  hard + medium 样本
  → 最终训练集
```

本模块只负责"生成"，`filter` 负责"筛选"，职责清晰。

---

## 输出结构

```
data_output/
├── bad_cases/
│   └── <Dataset>/
│       └── bad_cases.jsonl          # Stage 1：bad case + 维度标签
├── diagnosis_reports/
│   └── <Dataset>/<split>/
│       └── dimension_reports.jsonl  # Stage 2：维度诊断报告
├── synth_raw/
│   └── <Dataset>/
│       └── candidates_iter*.jsonl   # Stage 3：原始合成候选
└── synth_clean/
    └── <Dataset>/
        ├── synthetic.parquet        # Stage 4：去重后的干净数据
        └── dedupe_summary_iter*.jsonl  # 去重统计
```

---

## 模型分工

| 阶段 | 使用模型 | 原因 |
|---|---|---|
| Stage 2 诊断 | **强模型**（deepseek-v4-flash）| 需要跨样本归纳抽象，理解复杂认知操作 |
| Stage 3 合成 | **强模型**（deepseek-v4-flash）| 生成高质量、格式正确的新题目 |
| Stage 4 去重 | 无 LLM，纯算法 | MinHash 不需要模型 |

---

## 支持的数据集与维度

| 数据集 | 维度字段 | 典型维度 |
|---|---|---|
| ToMBench | `meta.ability` | `false_belief`, `perspective_taking`, `faux_pas` |
| BigToM | `meta.condition_type` | `true_belief`, `false_belief` |
| SocialIQA | `meta.dimension` | `xWant`, `xNeed`, `xReact`, `xEffect` |
| EmoBench | `meta.dimension` | 情绪归因相关维度 |
| FanToM | `meta.question_type` | 信念追踪相关维度 |
| HiToM | `meta.order` | `order_1`, `order_2`（信念阶数）|

---

## 入口命令

```bash
# 完整流水线
python run_feedback.py --stage all --dataset ToMBench

# 单阶段
python run_feedback.py --stage load --dataset BigToM --max-bad-cases 80
python run_feedback.py --stage diagnose --dataset BigToM
python run_feedback.py --stage synth
python run_feedback.py --stage dedupe
```

---

## 添加新数据集

1. 在 `config.yaml` 的 `synthesis_datasets` 中添加数据集配置
2. 在 `stage3_synthesis.py` 的 schema registry 中注册对应 Pydantic schema
3. 在 `stage1_load_predictions.py:get_dimension_key()` 中添加维度字段映射
