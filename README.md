# ToMEval 使用说明

面向心智理论（Theory-of-Mind）及相关社会认知基准的标准化 QA **评测 + 数据治理**框架。

围绕三条流水线，配合一套报告工具，全部用根目录的 `python run_*.py` 一键驱动，配置即用、无需改代码：

| 流水线 | 入口 | 配置 | 作用 |
|---|---|---|---|
| **评测** | `python run_eval.py` | `experiment_config.yaml` | 对模型在各数据集上跑预测 → 判分 → 出图 |
| **数据质量评估** | `python run_filter.py` | `filter/config.yaml` | 用决策树（pass@k / 可回答性 / shortcut）评估并自动修复数据 |
| **数据合成** | `python run_feedback.py` | `feedback/config.yaml` | 从模型 bad case 诊断薄弱维度 → 合成新训练数据 |
| 报告 | `python report/generate_*.py` | `report/tables_config.yaml` | 把评测结果汇总成 Markdown 表格 / HTML 报告 |

---

## 目录

- [安装](#安装)
- [标准数据格式](#标准数据格式)
- [一、评测流水线 `run_eval.py`](#一评测流水线-run_evalpy)
  - [第一步：配置模型](#第一步配置模型)
  - [第二步：运行评测](#第二步运行评测)
  - [第三步：查看结果](#第三步查看结果)
  - [第四步：可视化出图](#第四步可视化出图)
  - [第五步：生成对比表格](#第五步生成对比表格)
  - [接入不同模型](#接入不同模型)
- [二、数据质量评估 `run_filter.py`](#二数据质量评估-run_filterpy)
- [三、数据合成 `run_feedback.py`](#三数据合成-run_feedbackpy)
- [仓库结构](#仓库结构)
- [新增数据集 / 模型](#新增数据集--模型)
- [常见问题](#常见问题)
- [文档索引](#文档索引)

---

## 安装

```bash
pip install -r requirements.txt
```

框架支持任何兼容 OpenAI API 的模型（本地 vLLM、DeepSeek、OpenAI、通义千问等）。

---

## 标准数据格式

所有数据集在评测前都已**归一化**为统一 schema，存放在 `datasets/<数据集名>/`（parquet）：

```json
{
  "story": "背景故事文本",
  "question": "问题文本",
  "answer": {
    "correct_answers": ["正确答案"],
    "wrong_answers": ["错误选项1", "错误选项2"]
  },
  "meta": {
    "id": "可选的样本ID",
    "...": "其他分组字段（如 ability / dimension / lang）"
  }
}
```

- `correct_answers` 永远是字符串列表。
- `wrong_answers` 为空 → **开放题**（模型输出自由文本，由 judge 判分）。
- `wrong_answers` 非空 → **选择题**（模型输出选项字母，规则判分）。
- 题型自动判定：无干扰项且唯一正确答案 = `open`；多个正确答案 = `mcq_multi`；否则 = `mcq_single`。
- `mcq_grouped`（多问合并）：一个 prompt 内含多道子问题（如 EmoBench EU 的「情绪 + 原因」两问），由数据集的 `prepare_samples` 钩子预先打包，规则判分要求每个子问题各对才算整体对。详见 [docs/add_new_dataset.md](docs/add_new_dataset.md)。
- 数据集专属的分组字段放进 `meta`，可视化和分组指标会自动用到。

> **复刻原论文 prompt（可选）**：默认 prompt 由框架按协议统一生成；若某数据集需要忠实复刻原论文的题面排版 / system prompt，可在 `tasks/<数据集>/prompt.py` 提供 `build_prompt` / `build_system_prompt` / `prepare_samples` 钩子（约定式加载，缺省回退通用实现）。当前 19 个数据集中除 SocialMind 外均已接入 `prompt.py`，其中 EmoBench 额外用 `prepare_samples` 把 EU 子集合并为 `mcq_grouped` 多问。详见 [docs/add_new_dataset.md](docs/add_new_dataset.md)。

> 📚 **数据集总览**：19 个数据集各自考察什么、从哪个原始数据集**改造**而来、以及如何评测，集中整理在 **[docs/datasets.md](docs/datasets.md)**。

---

## 一、评测流水线 `run_eval.py`

整体流程：

```
配置 experiment_config.yaml
        ↓
  python run_eval.py            ← 预测 → 判分 → 出图（结果存 results/、图存 figures/）
        ↓
python report/generate_*.py     ← 汇总成 Markdown 表格 / HTML 报告（存 tables/）
```

框架自动完成：加载数据 →（可选）数据集 `prepare_samples` 预处理 → 按协议（或数据集自定义 `prompt.py`）构造 Prompt → 并发调用模型 → 判分（选择题 / grouped 多问走规则判分，开放题走 LLM judge）→ 汇总指标 → 多轮取平均 → 出图。

### 第一步：配置模型

所有评测配置都在根目录的 `experiment_config.yaml`：

```yaml
# ── 被测模型（只填连接相关参数）────────────────────────────
llm:
  model_name: MyModel        # 自定义名称，同时是 results/ 下的目录名
  api_key: not-needed        # 本地服务填 not-needed；云端 API 填真实 key（支持 ${ENV_VAR}）
  api_url: http://0.0.0.0:8000/v1
  max_workers: 16            # 并发线程数；云端 API 撞限流就调小（如 3~8）

# ── 评测协议（采样参数 / system prompt / extractor / 投票全部由协议驱动）──
protocol: cot                # direct | direct_think | cot | del_tom，详见 docs/protocols.md

# ── 评测阶段与数据集 ──────────────────────────────────────
stage: all                   # predict（只推理）| metric（只判分）| visualize（只出图）| all
datasets:                    # run_eval.py 批量评测的数据集列表
  - BigToM
  - EmoBench
  - FanToM
  - HiToM
  - SocialIQA
  - ToMBench

# ── 实验参数 ──────────────────────────────────────────────
max_samples: 0               # 0 = 跑全量；调试时可设 3~30（固定随机种子，可复现）
normalized_datasets_path: datasets   # 标准化数据集根目录
results_path: results        # 预测/判分结果根目录
figures_path: figures        # visualize 阶段出图根目录
```

> **关键点**
> - `model_name` 决定结果存哪个目录，不同模型必须用不同名称。
> - **温度 / max_tokens / 是否思考 / 重复次数（= 协议的 n_samples）全部由 `protocol` 决定**，`llm` 段只保留连接信息，避免手动设错。各协议的参数与 prompt 详见 [docs/protocols.md](docs/protocols.md)。
> - 选择题走规则判分（提取 `\boxed{}` 比对字母），**不需要 judge 模型**。开放题的判分方式由**数据集自己**在 `tasks/<数据集>/config.yaml` 的 `open_judge` 字段选择（`f1` / `llm_simple` / `rubric`），需要 judge 模型时在同一文件里配 `judge1`/`judge2`，与被测模型解耦。详见 [docs/add_new_dataset.md](docs/add_new_dataset.md)。

### 第二步：运行评测

```bash
# 跑全部数据集（datasets 列表）
python run_eval.py

# 只跑某一个数据集（同样从 experiment_config.yaml 读 protocol/stage）
python tasks/ToMBench/run.py
```

**分阶段运行**——改 `experiment_config.yaml` 里的 `stage` 再运行：

| stage | 作用 |
|---|---|
| `predict` | 只推理，产出 `prediction.jsonl`（适合先攒预测） |
| `metric` | 对已有预测重新判分（复用最新 exp 目录），产出 `metrics.json` |
| `visualize` | 按 `metrics.json` 出图到 `figures/<数据集>/<模型>/` |
| `all` | 预测 + 判分 + 出图，端到端 |

```bash
# metric / visualize 复用最新实验目录；也可用 --exp-dir 指定已有目录
python run_eval.py --exp-dir 20260516_230618
```

### 第三步：查看结果

```
results/
└── {数据集}/
    └── {model_name}/
        └── exp_{时间戳}/
            ├── config.json       # 本次实验配置（api_key/api_url 已自动脱敏）
            ├── metrics.json       # 汇总指标
            └── prediction.jsonl   # 每条样本的预测详情
```

**`metrics.json` 结构**：

```json
{
  "avg_metrics": {
    "accuracy": 0.7340, "correct": 743, "total": 1012,
    "by_ability": { "Belief: Content false beliefs": 0.71 },
    "per_sample_results": [ { "sample_id": "...", "is_correct": true, "error_reason": null, "prompt_type": "mcq_single" } ]
  },
  "all_metrics": [ { "accuracy": 0.7321, "correct": 741, "total": 1012, "per_sample_results": [ ... ] } ]
}
```

- `avg_metrics`：多轮平均指标，含各分组准确率（`by_*` 随数据集而异）。
- `all_metrics`：每一轮的原始指标 + 逐样本判分结果 `per_sample_results`。

**`prediction.jsonl` 结构**（每行一条，只存**原始模型输出**，判分在 metric 阶段做）：

```json
{
  "sample_id": "emo_0001",
  "sample_index": 0,
  "repeat": 0,
  "prompt_type": "mcq_single",
  "story": "...", "question": "...",
  "correct_answers": ["..."], "wrong_answers": ["..."],
  "options": {"A": "...", "B": "..."},
  "correct_letters": ["B"], "wrong_letters": ["A", "C"],
  "shuffle_seed": 123456,
  "prompt": "Read the following story...",
  "pred": { "content": "...\\boxed{B}", "reasoning": "..." },
  "meta": {"ability": "..."},
  "protocol": "cot", "extractor": "cot"
}
```

> `pred.content` 是模型的原始文本输出；`pred.reasoning` 是 thinking 模式下剥离出的思考过程。
> `is_correct` / `error_reason` 不在这里——它们是判分结果，落在 `metrics.json` 的 `per_sample_results`。

### 第四步：可视化出图

`visualize` 阶段（或 `python -m src.visualization`）是**数据集无关**的：它只读 `metrics.json` 里的 `by_*` 分组指标，自动出图到 `figures/<数据集>/<模型>/`：

- 每个 `by_*` 分组 → 柱状图（带样本数）；含 `|` 的复合键 → 热力图；
- 开放题的 rubric 打分 → 平均分柱状图；多 judge → 一致性散点 + Bland-Altman；
- 传入多个 `--results` → 额外出多模型对比图 / 雷达图。

```bash
# 单模型出图
python -m src.visualization --results results/SocialMind/Qwen3-8B/exp_xxxx --out figures/SocialMind
# 多模型对比（传多个 --results）
python -m src.visualization --results results/SocialMind/A/exp_xxx results/SocialMind/B/exp_yyy --out figures/SocialMind
```

详见 [docs/visualization.md](docs/visualization.md)。

### 第五步：生成对比表格

```bash
python report/generate_dataset_tables.py   # 各数据集详细表 → tables/{数据集}/基础指标.md + 其他指标.md
python report/generate_summary.py          # 跨数据集汇总表 → tables/SUMMARY.md
python report/generate_html_report.py      # 可选：交互式 HTML 报告（带热力图）→ tables/report.html
```

用 `report/tables_config.yaml` 控制进表的模型/数据集：

```yaml
results_dir: results
output_dir: tables
exp_suffix:                 # 不填则取最新一次实验
dataset:                    # 不填则处理所有数据集
  - ToMBench
models:                     # 不填则处理所有模型；display 是表里显示的名称
  - name: Qwen3-8B          # 与 experiment_config.yaml 的 model_name 一致
    display: Qwen3-8B-Think
```

> 对比多个模型：用不同 `model_name` 各跑一次评测，结果自动存进不同目录，生成表格时并排展示。

### 接入不同模型

**本地 vLLM**：

```bash
vllm serve /path/to/model --port 8000                                   # 普通模型
vllm serve /path/to/Qwen3-8B --port 8000 \
    --reasoning-parser qwen3                                            # 支持 thinking 的模型
```

```yaml
llm:
  model_name: Qwen3-8B
  api_key: not-needed
  api_url: http://0.0.0.0:8000/v1
# 是否走 thinking 由协议决定：direct 关闭、其余开启
```

确认服务正常：`curl http://0.0.0.0:8000/v1/models`

**云端 OpenAI 兼容 API**：

| 服务 | api_url | 常用 model_name |
|---|---|---|
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o` |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen3-8b` / `qwen-max` |
| 智谱 AI | `https://open.bigmodel.cn/api/paas/v4` | `glm-4` |
| 月之暗面 | `https://api.moonshot.cn/v1` | `moonshot-v1-8k` |

```yaml
llm:
  model_name: deepseek-chat
  api_key: ${DEEPSEEK_API_KEY}      # 或直接填 key（注意别提交到 git）
  api_url: https://api.deepseek.com/v1
  max_workers: 8                    # 云端并发建议降低，撞限流再调小
protocol: cot
```

> ⚠️ 部分云端 API 有约束：dashscope qwen3-8b 的 `max_tokens` 上限是 8192，且 thinking 开启时只能流式调用——配 judge 时记得在 `judge1` 里写 `enable_thinking: false` 和 `max_tokens: 8192`。

更多细节见 [docs/add_new_model.md](docs/add_new_model.md)。

---

## 二、数据质量评估 `run_filter.py`

`filter/` 是**数据质量评估 + 自动修复**流水线，用决策树逐层筛查训练数据里的低质题（太简单 / 无法回答 / 可走捷径），并迭代修复。

**评估链路（逐层缩小检测范围，省 token）：**

1. **pass@k**：用一个**弱模型**对每条样本独立预测 `pass_k` 次（默认 3）。全过 → `easy`（太简单）；全错 → 待定；部分过 → 进入下一层。
2. **可回答性**：用**强模型**判断 partial / 全错的样本是否自洽、答案是否唯一。判 false → `bad`。
3. **shortcut 三维探测**：对 partial 且可回答的样本，分别去掉故事 / 问题 / 选项各试 `shortcut_k` 次，看是否仍能蒙对。命中 → `shortcut`。
4. **决策树打标 + 修复**：综合上面结果给标签（`easy/hard/medium/shortcut/bad/unfixable`），对 `easy/bad/shortcut` 自动修复并迭代（最多 `max_iter` 轮，仍修不好记 `unfixable`）。
5. **finalize**：把所有轮次里的 `hard + medium` 合并成最终训练集。

### 配置 `filter/config.yaml`

```yaml
datasets: [BigToM, EmoBench, FanToM, HiToM, SocialIQA, ToMBench]

models:
  strong:                  # 强模型：可回答性判断 + 修复
    model_name: deepseek-v4-flash
    api_url: https://dashscope.aliyuncs.com/compatible-mode/v1
    api_key: <YOUR_API_KEY>
    max_tokens: 8192
  simple:                  # 弱模型：pass@k 评估
    model_name: qwen3-8b
    api_url: https://dashscope.aliyuncs.com/compatible-mode/v1
    api_key: <YOUR_API_KEY>
    max_tokens: 8192

paths:
  input_root: train_datasets      # 输入：train_datasets/<数据集>/*.parquet
  output_root: filter_output      # 输出根目录

pass_k: 3                  # pass@k 的 k
shortcut_k: 3             # shortcut 每维试探次数（独立于 pass_k）

# 阶段开关（按需关闭，优雅降级）
passk_enabled: true          # false = 跳过难度分桶，全部按 partial 走
answerability_enabled: true  # false = 跳过可回答性，假设全部可回答
shortcut_enabled: true       # false = 跳过 shortcut，假设全部非 shortcut
repair_enabled: true         # false = 只评估打标，不修复 / 不迭代
max_iter: 2               # 修复迭代上限（含第一轮）
```

### 运行

```bash
python run_filter.py          # 无命令行参数，全部从 filter/config.yaml 读
```

输出落在 `filter_output/`：各数据集的逐轮评估明细、`final/train_set.parquet`（最终训练集），以及顶层 `SUMMARY_REPORT.md` 汇总报告。详见 [filter/README.md](filter/README.md)。

---

## 三、数据合成 `run_feedback.py`

`feedback/` 是**数据合成**流水线：分析模型的错题，诊断薄弱的认知维度，针对性地合成新训练数据。

**四个阶段：**

1. **load**：从 `results/<数据集>/<模型>/exp_*/prediction.jsonl` 收集多个模型的 bad case 并集，按维度分层采样。
2. **diagnose**：按能力维度分组错题，让 LLM 总结常见错误模式、认知操作、建议的合成主题，产出诊断报告。
3. **synth**：依据诊断报告调用 LLM 生成新题目（输出与训练数据 parquet 同构）。
4. **dedupe**：用 MinHash LSH 过滤掉与**测试集**相似的样本（防泄漏）以及内部重复，写出最终 parquet。

### 配置 `feedback/config.yaml`

```yaml
stage: all                # all / load / diagnose / synth / dedupe
predictions_root: results # bad case 来源（ToMEval 评测结果）
output_path: feedback_output
max_bad_cases: 0          # 每数据集最多收集的 bad case 数（0=不限）

synthesis_model:          # 合成用的模型（建议强模型）
  model_name: deepseek-v4-flash
  api_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  api_key: <YOUR_API_KEY>

leakage_guard:            # LSH 防泄漏守门员
  test_root: datasets
  threshold: 0.6          # 与测试集相似度阈值
  internal_threshold: 0.85

synthesis_datasets:       # 要合成哪些数据集、各合成多少
  - name: ToMBench
    target_samples: 50
    samples_per_report: 5
    models:
      - name: Qwen3-8B
        exp: ~            # ~ = 自动取最新 exp_* 目录
```

### 运行

```bash
python run_feedback.py                          # 用默认 feedback/config.yaml
python run_feedback.py --config feedback/config.yaml   # 指定配置
```

最终合成数据落在 `feedback_output/`（清洗后的 parquet）+ `SUMMARY.md` 汇总。可改 `stage` 单独跑某一阶段。详见 [feedback/README.md](feedback/README.md)。

---

## 仓库结构

```text
ToMEval/
├── experiment_config.yaml     # 评测配置
├── run_eval.py                # 评测入口
├── run_filter.py              # 数据质量评估入口
├── run_feedback.py            # 数据合成入口
├── requirements.txt
├── src/
│   ├── evaluation/            # 评测核心
│   │   ├── pipeline.py        # 三阶段编排（predict/metric/visualize）
│   │   ├── data.py            # 标准化数据加载
│   │   ├── prediction.py      # 预测生成
│   │   ├── protocols.py       # 协议：采样参数 / system prompt / extractor
│   │   ├── prompt_loaders.py  # 按数据集约定加载 tasks/<DS>/prompt.py 的自定义钩子
│   │   ├── lang.py            # 样本语言归一化（meta.lang/language → en/zh）
│   │   ├── voting.py          # del_tom 多数投票
│   │   ├── judge.py           # 判分分派（MCQ / grouped 规则 + open 委托）
│   │   ├── open_judge.py      # 开放题判分模式（f1 / llm_simple / rubric）
│   │   ├── prompts.py / storage.py / paths.py / metrics.py / task_metrics.py / types.py
│   ├── llm/                   # OpenAI 兼容客户端（ContentClient / StructureClient）
│   ├── dataloader/            # parquet 数据加载
│   └── visualization/         # 数据集无关出图
├── tasks/<数据集>/            # 各数据集入口：config.yaml / run.py / metrics.py（可选 prompt.py 复刻原论文 prompt）
├── datasets/                  # 标准化后的测试数据集（parquet）
├── train_datasets/            # 训练数据集（filter 的输入）
├── filter/                    # 数据质量评估流水线（config.yaml / pipeline.py / eval/ / repair/）
├── feedback/                  # 数据合成流水线（config.yaml / stage1~4 / prompts.py）
├── report/                    # 报告生成（generate_dataset_tables / summary / html_report）
├── results/                   # 评测结果
├── figures/                   # 可视化输出
├── tables/                    # 生成的表格 / 报告
├── docs/                      # 详细文档
└── logs/                      # 日志
```

---

## 新增数据集 / 模型

- **新增数据集**：归一化数据 → 建 `tasks/<数据集>/{config.yaml, run.py, metrics.py}` → 加进 `experiment_config.yaml` 的 `datasets`。开放题数据集再在 `config.yaml` 配 `open_judge`。详见 [docs/add_new_dataset.md](docs/add_new_dataset.md)。
- **新增模型**：改 `experiment_config.yaml` 的 `llm` 段即可，无需改代码。详见 [docs/add_new_model.md](docs/add_new_model.md)。

---

## 常见问题

**Q：运行时报 "Connection refused"？**
检查本地 vLLM 是否在跑：`curl http://0.0.0.0:8000/v1/models`。

**Q：云端 API 报 429 限流 / 思考模式报错？**
调小 `max_workers`（如 3）；dashscope qwen3 thinking 开启时只支持流式，judge 这类非流式调用要设 `enable_thinking: false`，且 `max_tokens` 不超过 8192。

**Q：思考（thinking）模式怎么控制？**
由协议统一控制：`direct` 关闭（裸答），`direct_think` / `cot` / `del_tom` 开启；思考过程存在 `prediction.jsonl` 的 `pred.reasoning`。详见 [docs/protocols.md](docs/protocols.md)。

**Q：模型不支持结构化输出怎么办？**
`StructureClient` 自动降级：先试原生 `parse` 模式，失败则改用「Prompt 注入 JSON 格式 + 正则提取」的 `create` 模式，无需手动干预。

**Q：随机抽样（`max_samples > 0`）每次结果一样吗？**
一样，内部使用固定随机种子，相同 `max_samples` 抽到的样本完全一致。

---

## 文档索引

| 文档 | 内容 |
|---|---|
| [docs/vision.md](docs/vision.md) | **项目愿景**：从 ToM 专用扩展到任意测试集的全自动迭代式数据合成框架（三原则：自动化 / 轻量化 / 可视化） |
| [docs/module_reference.md](docs/module_reference.md) | **模块复用参考手册**：每个模块用途、关键函数、import 路径、跨模块复用速查（写新流程前先查这里） |
| [docs/refactor_backlog.md](docs/refactor_backlog.md) | **冗余清理 backlog**：已识别的重复代码/冗余文件清单，按优先级排序，供"每10个任务一次清理"使用 |
| [docs/protocols.md](docs/protocols.md) | 四个评测协议的采样参数、system prompt、extractor、投票、shuffle |
| [docs/visualization.md](docs/visualization.md) | 可视化模块用法与自动出图行为 |
| [docs/add_new_dataset.md](docs/add_new_dataset.md) | 如何新增数据集（含开放题 `open_judge` 配置） |
| [docs/add_new_model.md](docs/add_new_model.md) | 如何接入新模型、查看结果、生成报告 |
| [docs/generate_tables.md](docs/generate_tables.md) / [docs/generate_summary.md](docs/generate_summary.md) | 报告表格生成 |
| [filter/README.md](filter/README.md) | 数据质量评估流水线细节 |
| [feedback/README.md](feedback/README.md) | 数据合成流水线细节 |
