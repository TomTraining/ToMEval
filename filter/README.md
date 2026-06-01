# Data Eval 使用指南

V3 数据飞轮：决策树评估 + 三维 shortcut 探测 + 自动修复迭代

## 快速开始

### 1. 准备数据

将待评估的数据集放到 `train_datasets/` 目录下：

```
train_datasets/
├── BigToM/
│   ├── train.parquet
│   └── test.parquet
├── EmoBench/
│   └── synthetic.parquet
└── ...
```

**要求：**
- 每个数据集一个子目录
- 目录下可以有任意数量的 `.parquet` 文件（会自动合并）
- 数据必须符合标准 schema（见下文）

### 2. 配置模型

编辑 `filter/config.yaml`：

```yaml
# 要评估哪些数据集
datasets:
  - BigToM
  - EmoBench
  - FanToM

# 模型配置
models:
  strong:  # 用于 answerability 判断和 repair
    api_url: https://your-api-url
    api_key: your-api-key
    model_name: deepseek-v4-flash
  simple:  # 用于 pass@k 评估
    model_name: qwen3-8b

# 采样控制（可选）
sampling:
  enabled: true      # false=全量评估
  max_samples: 100   # 每个数据集最多评估多少条

# 修复迭代次数
max_iter: 3
```

### 3. 运行评估

```bash
python run_eval.py
```

就这么简单！程序会自动：
1. 对每个数据集执行 **filter**（决策树评估 + 修复迭代）
2. 执行 **finalize**（合并 hard+medium → train_set.parquet）

### 4. 查看结果

```
filter_output/
├── BigToM/
│   ├── eval_iter1/
│   │   ├── passk.parquet          # pass@k 结果
│   │   ├── answerability.parquet  # 可答性判断
│   │   ├── shortcut.parquet       # shortcut 探测
│   │   ├── labels.parquet         # 决策树标签
│   │   ├── repaired.parquet       # 修复后的样本
│   │   └── summary.json           # 本轮统计
│   ├── eval_iter2/
│   │   └── ...
│   ├── filter_summary.json        # 全流程统计
│   └── final/
│       └── train_set.parquet      # 最终训练集（hard+medium）
└── ...
```

**日志文件：** `logs/filter_{timestamp}.log`

---

## 数据 Schema

输入数据必须包含以下字段：

```python
{
  "story": str,              # 上下文/故事
  "question": str,           # 问题
  "answer": {
    "correct_answers": [str],   # 正确答案列表
    "wrong_answers": [str]      # 错误答案列表（可为空）
  },
  "meta": {                  # 可选元数据
    "id": str,               # 样本 ID（推荐）
    ...
  }
}
```

**题型判断：**
- `wrong_answers` 为空 → **开放问答**（模型输出自由文本）
- `wrong_answers` 非空 + `len(correct_answers) == 1` → **单选题**
- `wrong_answers` 非空 + `len(correct_answers) > 1` → **多选题**

---

## 评估流程

### Phase B: Pass@K 评估
- 用弱模型（simple）对每个样本做 k=3 次预测
- 根据通过次数分为三类：
  - **all_passed** (3/3) → 太简单
  - **partial** (1-2/3) → 中等难度
  - **all_failed** (0/3) → 太难或有问题

### Phase C: Answerability 判断
- 仅对 `partial` 和 `all_failed` 样本执行
- 用强模型（strong）判断样本是否可答：
  - 前提是否自洽
  - 答案是否唯一
  - 逻辑是否无误

### Phase D: Shortcut 探测
- 仅对 `partial + answerable` 样本执行
- 三维探测：
  - **no_story**: 去掉 story 能否答对
  - **no_question**: 去掉 question 能否答对
  - **no_options**: 去掉选项文本（仅保留字母）能否答对
- 如果任意维度通过 ≥ majority (ceil(k/2))，标记为 shortcut

### Phase E: 决策树标签 + 修复
根据上述结果打标签：
- **easy** (all_passed) → 修复：增加难度
- **hard** (all_failed + answerable) → **保留**
- **medium** (partial + answerable + not shortcut) → **保留**
- **shortcut** (partial + answerable + is_shortcut) → 修复：消除 shortcut
- **bad** (unanswerable) → 修复：修正逻辑错误
- **unfixable** (max_iter 后仍未修复) → 丢弃

修复后的样本进入下一轮迭代，直到：
- 达到 `max_iter` 上限
- 无新样本需要修复

### Finalize: 合并训练集
收集所有轮次中标记为 `hard` 和 `medium` 的样本，合并为最终训练集。

---

## 配置说明

### 固定参数（硬编码，无需配置）
- `pass_k = 3` - pass@k 的 k 值
- `shortcut_threshold = "majority"` - ceil(k/2) = 2
- `shortcut_dimensions = [no_story, no_question, no_options]`
- `skip_answerability_on_all_passed = true` - all_passed 直接判 easy
- `repair_enabled_types = [unanswerable, easy, shortcut]`

### 可调参数（config.yaml）
- `datasets` - 数据集列表
- `models.strong` / `models.simple` - 模型配置
- `paths.input_root` / `paths.output_root` - 路径
- `sampling.enabled` / `sampling.max_samples` - 采样控制
- `max_iter` - 修复迭代上限

---

## 常见问题

### Q: 为什么我的数据集被跳过？
A: 检查 `train_datasets/{dataset}/` 下是否有 `.parquet` 文件。

### Q: 如何只评估部分数据集？
A: 修改 `config.yaml` 中的 `datasets` 列表。

### Q: 如何关闭采样？
A: 设置 `sampling.enabled: false`。

### Q: 修复迭代太慢怎么办？
A: 减少 `max_iter`，或开启采样减少样本数。

### Q: 如何查看详细日志？
A: 查看 `logs/filter_{timestamp}.log` 文件。

---

## 设计文档

完整设计见：
- [V3 数据飞轮设计](../docs/data_flywheel_v3.md)
- [评估流程可视化](./docs/OVERVIEW.md)
- [评估模块说明](./eval/README.md)
- [修复模块说明](./repair/README.md)
