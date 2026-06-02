# Data Eval 模块概览

> **一句话目标**：把合成的 ToM 训练数据从"生成即用"变成"质量可控"——自动识别太简单、有捷径、不可答的样本，修复后再评，最终只保留真正有训练价值的 hard / medium 样本。

---

## 为什么需要这个模块

合成数据天然存在三类质量问题：

| 问题类型 | 表现 | 危害 |
|---|---|---|
| **太简单（easy）** | 弱模型 3 次全答对 | 训练无增益，浪费算力 |
| **有捷径（shortcut）** | 去掉 story / question / options 仍能答对 | 模型学到表面规律而非推理能力 |
| **不可答（bad）** | 前提矛盾、答案歧义、逻辑错误 | 引入噪声，损害模型对齐 |

直接用这些数据训练，模型学到的是"怎么猜"而不是"怎么推理"。

---

## 核心设计：决策树 + 迭代修复

不是对每条样本做全量评估，而是**按结果分流**，每条样本只走自己该走的路径，节省调用量。

```
输入数据（synthetic.parquet）
        │
        ▼
  Phase B：pass@k
  弱模型跑 k=3 次
        │
   ┌────┴────────────────────┐
   ▼                         ▼
all_passed (3/3)         partial (1-2/3) 或 all_failed (0/3)
   │                         │
   ▼                         ▼
标 easy → 修复          Phase C：answerability
（提难度）              强模型判断是否可答
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
               unanswerable       answerable
                    │                 │
                    ▼                 ├── all_failed → 标 hard（保留）
               标 bad → 修复          │
               （修逻辑）             ▼
                              Phase D：shortcut 三探测
                              去 story / 去 question / 去 options
                                      │
                             ┌────────┴────────┐
                             ▼                 ▼
                        任一维度异常        三维度正常
                             │                 │
                             ▼                 ▼
                        标 shortcut        标 medium（保留）
                        → 修复
                        （重写 story）
```

修复后的样本进入**下一轮迭代**，直到达到 `max_iter` 上限或无新样本需要修复。

---

## 六种样本标签

| 标签 | 条件 | 去向 |
|---|---|---|
| **hard** | all_failed + answerable | ✅ 保留进训练池 |
| **medium** | partial + answerable + 无 shortcut | ✅ 保留进训练池 |
| **easy** | all_passed | 修复：提高难度 |
| **shortcut** | partial + answerable + 有 shortcut | 修复：重写 story |
| **bad** | unanswerable（任意 pass@k 结果）| 修复：修正逻辑/标签 |
| **unfixable** | 达到 max_iter 仍未修复 | ❌ 丢弃 |

---

## Shortcut 三维探测

单一维度（如"换个错误 story"）只能发现一种捷径。V3 扩展到三个维度：

| 探测维度 | 做法 | 发现的问题 |
|---|---|---|
| **no_story** | 去掉 story，只给 question + options | 答案藏在问题本身，不需要推理上下文 |
| **no_question** | 去掉 question，只给 story + options | 选项本身暴露了答案 |
| **no_options** | 去掉选项文本，只保留字母 | 靠排除法而非理解作答 |

任意一个维度：弱模型 k 次中过半答对 → 标记为 shortcut。

---

## 迭代闭环

```
iter 1: 评估原始数据 → 标签 → 修复 → repaired_1.parquet
iter 2: 评估 repaired_1 → 标签 → 修复 → repaired_2.parquet
...
iter N: 评估 repaired_{N-1} → 标签（无修复）
finalize: 收集所有轮次的 hard + medium → train_set.parquet
```

每轮只重评"上轮修复的样本"，已经是 hard / medium 的样本不重复评估。

---

## 输出结构

```
filter_output/<Dataset>/
├── eval_iter1/
│   ├── passk.parquet          # Phase B 结果（pass_count, bucket）
│   ├── answerability.parquet  # Phase C 结果（is_answerable, reason）
│   ├── shortcut.parquet       # Phase D 结果（三维探测明细）
│   ├── labels.parquet         # 决策树最终标签
│   ├── repaired.parquet       # 修复后样本（进入下一轮）
│   └── summary.json           # 本轮各标签占比统计
├── eval_iter2/
│   └── ...
├── filter_summary.json        # 全流程汇总
└── final/
    └── train_set.parquet      # 最终训练集（hard + medium 合并）
```

---

## 模型分工

| 阶段 | 使用模型 | 原因 |
|---|---|---|
| Phase B pass@k | **弱模型**（qwen3-8b）| 用弱模型测难度才有区分度；强模型全过，信息量为零 |
| Phase C answerability | **强模型**（deepseek-v4-flash）| 需要理解复杂逻辑，判断前提是否自洽 |
| Phase D shortcut | **弱模型** | 探测"弱模型能否不靠推理答对"，用弱模型才有意义 |
| Phase E repair | **强模型** | 生成修复后的高质量样本 |

---

## 与 V2 的核心差异

V2 是串行流水线（answerability 全量 → pass@k → shortcut 单维度），V3 改为决策树：

- **answerability 从全量变为按需**：all_passed 样本直接判 easy，不浪费强模型调用
- **shortcut 从单维度变为三维度**：覆盖更多捷径模式
- **shortcut 触发范围扩大**：V2 只对 all_passed 做，V3 对 partial + answerable 做（all_failed 不可能靠 shortcut 答对，跳过）

---

## 入口命令

```bash
# 完整流程（评估 + 修复迭代 + 合并）
python run_filter.py

# 只跑评估+修复，不合并
python run_filter.py --eval filter --dataset BigToM --max-iter 3

# 只合并最终训练集
python run_filter.py --eval finalize --dataset BigToM
```
