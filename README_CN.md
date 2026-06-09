# ToMEval 使用说明

面向心智理论（Theory-of-Mind）及相关社会认知基准的标准化 QA 评测框架。

---

## 目录

- [整体流程](#整体流程)
- [快速开始](#快速开始)
- [第一步：配置模型](#第一步配置模型)
- [第二步：运行评测](#第二步运行评测)
- [第三步：查看结果](#第三步查看结果)
- [第四步：生成对比表格](#第四步生成对比表格)
- [对比多个模型](#对比多个模型)
- [接入不同模型](#接入不同模型)
- [常见问题](#常见问题)
- [新增数据集](#新增数据集)

---

## 整体流程

```
配置 experiment_config.yaml
        ↓
  python run_eval.py         ← 预测 + 打分，结果存 results/
        ↓
python report/generate_*.py  ← 生成 Markdown 表格 / HTML 报告，存 tables/
```

框架会自动完成：加载数据 → 构造 Prompt → 并发调用模型 → LLM Judge 打分 → 汇总指标 → 多轮取平均。**你只需要改配置文件，不需要动代码。**

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 修改配置（见下方说明）
vim experiment_config.yaml

# 3. 运行全部数据集
python run_eval.py

# 4. 生成报告
python report/generate_dataset_tables.py
python report/generate_summary.py
python report/generate_html_report.py   # 可选：生成 HTML 可视化
```

---

## 第一步：配置模型

所有配置都在根目录的 `experiment_config.yaml`。

```yaml
# ── 被测模型（只填连接相关参数）────────────────────────────
llm:
  model_name: MyModel        # 自定义名称，同时是 results/ 下的目录名
  api_key: not-needed        # 本地服务填 not-needed；云端 API 填真实 key
  api_url: http://0.0.0.0:8000/v1
  max_workers: 16            # 并发线程数，云端 API 建议降低到 8-16

# ── 评测协议（采样参数 / system prompt / extractor / 投票全部由协议驱动）──
protocol: cot                # direct | direct_think | cot | del_tom，详见 docs/protocols.md

# ── 评测阶段与数据集 ──────────────────────────────────────
stage: all                   # predict（只推理）| metric（只判分）| all
datasets:                    # run_eval.py 批量评测的数据集列表
  - BigToM
  - EmoBench
  - FanToM
  - HiToM
  - SocialIQA
  - ToMBench

# ── 实验参数 ──────────────────────────────────────────────
max_samples: 0               # 0 = 跑全量；调试时可设 3~10
normalized_datasets_path: datasets
results_path: results
```

> **关键点**
> - `model_name` 决定结果存在哪个目录，不同模型必须用不同名称
> - `api_key` 支持环境变量：`api_key: ${DEEPSEEK_API_KEY}`
> - **温度、max_tokens、是否思考、重复次数（= 协议的 n_samples）全部由 `protocol` 决定**，`llm` 段只保留连接信息，避免手动设错。各协议的参数与 prompt 详见 [docs/protocols.md](docs/protocols.md)
> - 当前数据集均为选择题，走规则判分（`\boxed{}` 提取），不需要 judge 模型；若日后加入开放题，judge 会自动回退使用 `llm` 配置（也可单独写 `judge:` 段覆盖）

---

## 第二步：运行评测

### 跑全部数据集（推荐）

```bash
python run_eval.py
```

跑哪些数据集由 `experiment_config.yaml` 的 `datasets` 列表决定（默认 6 个：`BigToM` / `EmoBench` / `FanToM` / `HiToM` / `SocialIQA` / `ToMBench`）。

### 只跑某一个数据集

```bash
python tasks/ToMBench/run.py
python tasks/SocialIQA/run.py
```

### 调试：先用少量样本验证配置

```yaml
# experiment_config.yaml
max_samples: 3
```

```bash
python tasks/ToMBench/run.py
```

没有报错再改回全量配置。

### 分阶段运行

在 `experiment_config.yaml` 里设置 `stage`，再运行 `python run_eval.py`：

```yaml
# 只跑推理（不打分），适合先攒好预测结果
stage: predict
```

```yaml
# 对已有的预测重新打分（不重新推理）
stage: metric
```
```bash
python run_eval.py --exp-dir 20260516_230618   # metric 阶段可指定已有实验目录
```

---

## 第三步：查看结果

评测完成后，结果保存在 `results/` 目录：

```
results/
└── {DatasetName}/
    └── {model_name}/
        └── exp_{timestamp}/
            ├── config.json        # 本次实验配置（API key 已自动脱敏）
            ├── metrics.json       # 汇总指标
            └── prediction.jsonl   # 每条样本的预测详情
```

### metrics.json 结构

```json
{
  "avg_metrics": {
    "accuracy": 0.7340,
    "correct": 743,
    "total": 1012,
    "by_ability": {
      "Belief: Content false beliefs": 0.71,
      "Knowledge: Percepts-knowledge links": 0.80
    }
  },
  "all_metrics": [
    {"accuracy": 0.7321, "correct": 741, "total": 1012},
    {"accuracy": 0.7343, "correct": 743, "total": 1012},
    {"accuracy": 0.7356, "correct": 744, "total": 1012}
  ]
}
```

- `avg_metrics`：多轮的平均指标，含各分组准确率（按数据集不同而异）
- `all_metrics`：每一轮的原始指标

---

## 第四步：生成对比表格

```bash
# 生成各数据集详细表格 → tables/{Dataset}/基础指标.md + 其他指标.md
python report/generate_dataset_tables.py

# 生成跨数据集汇总表 → tables/SUMMARY.md
python report/generate_summary.py

# 生成可交互 HTML 报告（带颜色热力图）→ tables/report.html
python report/generate_html_report.py
```

### 控制哪些模型/数据集进入表格

编辑 `report/tables_config.yaml`：

```yaml
results_dir: results
output_dir: tables

# 不填则取最新一次实验
exp_suffix:

# 不填则处理所有数据集
dataset:
  - ToMBench
  - SocialIQA

# 不填则处理所有模型；display 是表格里显示的名称
models:
  - name: Qwen3-8B        # 与 experiment_config.yaml 里的 model_name 一致
    display: Qwen3-8B-Think
  - name: deepseek-chat
    display: DeepSeek-Chat
```

---

## 对比多个模型

分别用不同的 `model_name` 各跑一次，结果会自动存在不同目录，生成表格时并排展示。

```yaml
# 第一次跑，model_name: Qwen3-8B
# 第二次跑，改为 model_name: deepseek-chat
```

```bash
python report/generate_dataset_tables.py   # 两个模型并排出现在表格里
python report/generate_summary.py
```

---

## 接入不同模型

框架支持任何兼容 OpenAI API 的服务。

### 本地 vLLM

```bash
# 启动 vLLM（普通模型）
vllm serve /path/to/model --port 8000

# 启动 vLLM（Qwen3 等支持 thinking 的模型）
vllm serve /path/to/Qwen3-8B --port 8000 \
    --enable-reasoning --reasoning-parser deepseek_r1
```

```yaml
llm:
  api_url: http://0.0.0.0:8000/v1
  api_key: not-needed
# 是否走 thinking 由协议决定：direct 关闭、其余开启（详见 docs/protocols.md）
```

确认服务正常：`curl http://0.0.0.0:8000/v1/models`

### 云端 API

| 服务 | api_url | 常用 model_name |
|---|---|---|
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o` |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-max` |
| 智谱 AI | `https://open.bigmodel.cn/api/paas/v4` | `glm-4` |
| 月之暗面 | `https://api.moonshot.cn/v1` | `moonshot-v1-8k` |

云端 API 示例：

```yaml
llm:
  model_name: deepseek-chat
  api_key: ${DEEPSEEK_API_KEY}      # 或直接填 key
  api_url: https://api.deepseek.com/v1
  max_workers: 16                   # 云端并发建议降低
protocol: cot                       # 采样参数由协议决定
```

```bash
export DEEPSEEK_API_KEY="sk-xxx"
python run_eval.py
```

---

## 常见问题

**Q: 运行时报 "Connection refused"**
检查 vLLM 是否在跑：`curl http://0.0.0.0:8000/v1/models`

**Q: 模型不支持结构化输出怎么办？**
框架自动检测并降级：先尝试原生 `parse` 模式，失败则改用在 Prompt 里注入 JSON 格式要求 + 正则提取，无需手动干预。

**Q: 思考（thinking）模式怎么控制？**
现在由协议统一控制：`direct` 关闭 thinking（裸答），`direct_think` / `cot` / `del_tom` 开启。开启后对 Qwen3 等模型会触发 Chain-of-Thought 推理，思考过程保存在 `prediction.jsonl` 的 `pred.reasoning` 字段里。详见 [docs/protocols.md](docs/protocols.md)。

**Q: 能否跳过已跑过的数据集，只补跑新增的？**
手动运行单个数据集即可：`python tasks/NewDataset/run.py`

**Q: 随机抽样（`max_samples > 0`）每次结果一样吗？**
是的，内部使用固定随机种子，相同 `max_samples` 每次抽取的样本完全一致。

---

## 新增数据集

详见 [docs/add_new_dataset.md](docs/add_new_dataset.md)，简要步骤：

1. 将数据集归一化为标准格式，放到 `datasets/<DatasetName>/`
2. 创建 `tasks/<DatasetName>/config.yaml`（填数据集名和路径）
3. 创建 `tasks/<DatasetName>/run.py`（一行调用，复制其他任务即可）
4. 创建 `tasks/<DatasetName>/metrics.py`（实现 `compute_metrics()`，只需准确率时直接复制简单版本）
5. 在 `experiment_config.yaml` 的 `datasets` 列表里加上数据集名称

### 标准数据格式

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
    "...": "其他分组字段"
  }
}
```

- `wrong_answers` 为空 → 开放式问答（模型输出自由文本）
- `wrong_answers` 非空 → 选择题（模型输出选项字母）

数据集存放在 `datasets/<DatasetName>/` 目录下。
