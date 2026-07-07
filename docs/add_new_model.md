# 新增模型测试指南

本指南介绍如何在 ToMEval 框架中使用新模型进行评测，以及评测后如何查看结果。

## 目录

- [概述](#概述)
- [步骤 1：配置 experiment_config.yaml](#步骤-1配置-experiment_configyaml)
- [步骤 2：运行评测](#步骤-2运行评测)
- [步骤 3：查看原始结果](#步骤-3查看原始结果)
- [步骤 4：生成表格和汇总](#步骤-4生成表格和汇总)
- [模型接入方式](#模型接入方式)
  - [本地 vLLM 部署](#本地-vllm-部署)
  - [云端 OpenAI 兼容 API](#云端-openai-兼容-api)
- [配置参数详解](#配置参数详解)
- [常见问题](#常见问题)

---

## 概述

ToMEval 支持任何兼容 OpenAI API 的模型（包括本地 vLLM、DeepSeek、OpenAI、通义千问等）。

测试新模型只需：

1. **修改** `experiment_config.yaml` 填写模型 API 信息
2. **运行** 评测脚本
3. **生成** 结果表格和汇总

无需修改任何其他代码。

---

## 步骤 1：配置 experiment_config.yaml

`experiment_config.yaml` 位于 ToMEval 根目录，是所有实验的统一配置文件。

```yaml
# ============================
# 被测模型（LLM）配置：只填连接相关参数
# ============================
llm:
  model_name: MyNewModel        # 模型名称，同时作为 results/ 下的目录名
  api_key: not-needed           # API 密钥（本地服务填 not-needed，云端 API 填真实密钥）
  api_url: http://0.0.0.0:8000/v1   # API 端点
  max_workers: 16               # 并发线程数（云端 API 建议 8-16）

# ============================
# 评测协议：采样参数 / system prompt / extractor / 投票全部由协议驱动
# ============================
protocol: cot                   # direct | direct_think | cot | del_tom，详见 docs/protocols.md

# ============================
# 评测阶段与数据集
# ============================
stage: all                      # predict | metric | visualize | all
datasets:                       # run_eval.py 批量评测的数据集列表
  - BigToM
  - EmoBench
  - FanToM
  - HiToM
  - SocialIQA
  - ToMBench

# ============================
# 实验参数
# ============================
max_samples: 0      # 0 = 全量；>0 = 随机抽取指定数量（用于快速测试）
normalized_datasets_path: datasets
results_path: results           # 预测/判分结果根目录
figures_path: figures           # visualize 阶段出图根目录
```

### 重要说明

- `model_name` 的值会直接作为 `results/{dataset}/{model_name}/` 的目录名，**不同模型必须使用不同名称**。
- `api_key` 支持环境变量写法：`api_key: ${DEEPSEEK_API_KEY}`，运行时自动替换。
- **温度、max_tokens、是否思考、重复次数（= 协议的 n_samples）全部由 `protocol` 决定**，`llm` 段不再写这些采样参数。各协议的具体参数与 prompt 见 [protocols.md](protocols.md)。
- 选择题走规则判分（提取 `\boxed{}` 比对字母），**不需要 judge 模型**。开放题的判分方式由**数据集自己**在 `tasks/<数据集>/config.yaml` 的 `open_judge` 字段选择（`f1` / `llm_simple` / `rubric`），需要 judge 模型时在同一文件配 `judge1`/`judge2`，与被测模型 `llm` 段解耦。详见 [add_new_dataset.md](add_new_dataset.md)。

---

## 步骤 2：运行评测

### 运行全部数据集

```bash
cd /path/to/ToMEval
python run_eval.py
```

这会依次运行 `experiment_config.yaml` 中 `datasets` 列表里所有启用的数据集（共 19 个，覆盖 ToM / 社会认知 / 语用 + 常识对照，清单见 [docs/datasets.md](datasets.md)）。

### 运行单个数据集

```bash
python tasks/ToMBench/run.py
python tasks/HiToM/run.py
python tasks/SocialIQA/run.py
```

### 快速冒烟测试

先用少量样本验证配置是否正确：

```yaml
# experiment_config.yaml
max_samples: 3   # 只取 3 条样本（repeats 由协议决定，无需手动设）
```

```bash
python tasks/ToMBench/run.py
```

验证成功后再改回全量配置。

### 运行进度

运行时会实时打印进度：

```
已加载 1012 条样本，数据集: ToMBench/test
开始推理，共 1012 个请求（1012 样本 × 1 轮）...
第 1 轮: accuracy=0.7321  (741/1012)
平均 accuracy: 0.7321
结果已保存到: results/ToMBench/MyNewModel/exp_20260422_143022/
```

> 上面对应 `protocol: cot`（1 轮）。轮数由协议决定：`direct` / `direct_think` / `cot` 各 1 轮，`del_tom` 8 轮并按字母多数投票，此时会打印多轮 accuracy 与 `平均 accuracy ± 标准差`。

---

## 步骤 3：查看原始结果

评测完成后，结果保存在 `results/` 目录：

```
results/
└── {DatasetName}/
    └── {model_name}/
        └── exp_{timestamp}/
            ├── config.json       # 完整实验配置（API key 已自动脱敏）
            ├── metrics.json      # 汇总指标
            └── prediction.jsonl  # 每条样本的详细预测记录
```

### 查看汇总指标

```bash
# 查看某个数据集的指标
cat results/ToMBench/MyNewModel/exp_*/metrics.json | python -m json.tool

# 快速查看 accuracy
python -c "
import json, glob
for f in glob.glob('results/*/*/exp_*/metrics.json'):
    data = json.load(open(f))
    acc = data.get('avg_metrics', {}).get('accuracy', '-')
    print(f'{f}: {acc}')
"
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

### prediction.jsonl 结构

每行是一个 JSON 对象，**只记录原始模型输出**（判分在 metric 阶段做，不写在这里）：

```json
{
  "sample_id": "tombench_0001",
  "sample_index": 0,
  "repeat": 0,
  "prompt_type": "mcq_single",
  "correct_letters": ["B"],
  "options": {"A": "...", "B": "..."},
  "prompt": "Read the following story...",
  "pred": {
    "content": "...\\boxed{B}",
    "reasoning": "Let me think step by step..."
  },
  "meta": {"ability": "Belief: Content false beliefs"},
  "protocol": "cot",
  "extractor": "cot"
}
```

> `pred.content` 是模型的**原始文本输出**（不是已解析的对象）；`pred.reasoning` 是 thinking 模式下剥离出的思考过程。
> `is_correct` / `error_reason` 是判分结果，落在 `metrics.json` 的 `per_sample_results`，不在 `prediction.jsonl` 里。

---

## 步骤 4：生成表格和汇总

评测完成后，使用 `report/` 下的脚本生成对比表格：

```bash
cd /path/to/ToMEval

# 4.1 生成各数据集详细表格（写入 tables/ 目录）
python report/generate_dataset_tables.py

# 4.2 生成跨数据集汇总表（写入 tables/SUMMARY.md）
python report/generate_summary.py
```

详细用法参见 [generate_tables.md](generate_tables.md) 和 [generate_summary.md](generate_summary.md)。

---

## 模型接入方式

### 本地 vLLM 部署

#### 启动 vLLM 服务

```bash
# 基础启动
vllm serve /path/to/model \
    --port 8000 \
    --tensor-parallel-size 1

# 启用思考模式（Qwen3 等支持 thinking 的模型）
vllm serve /path/to/Qwen3-8B \
    --port 8000 \
    --reasoning-parser qwen3

# 多 GPU
vllm serve /path/to/model \
    --port 8000 \
    --tensor-parallel-size 4
```

#### 对应配置

```yaml
llm:
  model_name: Qwen3-8B          # 与模型目录名一致（或自定义）
  api_key: not-needed
  api_url: http://0.0.0.0:8000/v1
protocol: cot                   # 是否走 thinking 由协议决定
```

#### 验证服务正常

```bash
curl http://0.0.0.0:8000/v1/models
```

---

### 云端 OpenAI 兼容 API

#### DeepSeek

```yaml
llm:
  model_name: deepseek-chat
  api_key: ${DEEPSEEK_API_KEY}
  api_url: https://api.deepseek.com/v1
  max_workers: 16      # 云端 API 建议降低并发
protocol: cot          # 采样参数由协议决定
```

```bash
export DEEPSEEK_API_KEY="sk-xxx"
python run_eval.py
```

#### OpenAI

```yaml
llm:
  model_name: gpt-4o
  api_key: ${OPENAI_API_KEY}
  api_url: https://api.openai.com/v1
  max_workers: 8
protocol: cot          # 采样参数由协议决定
```

```bash
export OPENAI_API_KEY="sk-xxx"
python run_eval.py
```

#### 其他兼容服务

| 服务 | api_url | 常用模型名 |
|---|---|---|
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o`, `gpt-4o-mini` |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-max`, `qwen-plus` |
| 智谱 AI | `https://open.bigmodel.cn/api/paas/v4` | `glm-4`, `glm-4-flash` |
| 月之暗面 | `https://api.moonshot.cn/v1` | `moonshot-v1-8k` |

---

## 配置参数详解

### llm 配置（只保留连接相关参数）

| 参数 | 说明 | 建议值 |
|---|---|---|
| `model_name` | 模型名称，也是 `results/` 下的目录名 | 使用清晰可辨的名称 |
| `api_url` | API 端点，末尾不要加 `/` | - |
| `api_key` | API 密钥，支持 `${ENV_VAR}` 变量替换 | - |
| `max_workers` | 并发线程数 | `16`（云端 8-16，本地可更高） |

> `temperature` / `max_tokens` / `top_p` / `enable_thinking` 不再写在 `llm` 段，**由 `protocol` 统一覆盖**（缺省时 `LLMClient` 仍有内置默认值兜底，但正常评测应交给协议）。各协议取值见 [protocols.md](protocols.md)。

### 协议 / 阶段 / 数据集

| 参数 | 说明 | 取值 |
|---|---|---|
| `protocol` | 评测协议，决定采样参数、system prompt、extractor、是否投票 | `direct` / `direct_think` / `cot` / `del_tom` |
| `stage` | 评测阶段 | `predict` / `metric` / `visualize` / `all` |
| `datasets` | `run_eval.py` 批量评测的数据集名称列表 | 数据集名数组 |

### 实验参数

| 参数 | 说明 | 建议值 |
|---|---|---|
| `max_samples` | 最大样本数，`0` 为全量 | `0`（正式），`3-10`（调试） |
| `normalized_datasets_path` | 标准化数据集目录 | `datasets` |
| `results_path` | 预测/判分结果输出目录 | `results` |
| `figures_path` | visualize 阶段出图目录 | `figures` |

> 重复次数（repeats）= 协议的 `n_samples`（direct/direct_think/cot 为 1，del_tom 为 8），不再单独配置。
> 选择题走规则判分，不需要 judge 模型；开放题的判分方式与 judge 模型由数据集在 `tasks/<数据集>/config.yaml` 的 `open_judge` / `judge1` 配置，详见 [add_new_dataset.md](add_new_dataset.md)。

---

## 常见问题

### Q: 运行时提示 "Connection refused"

检查 vLLM 服务是否正在运行：

```bash
curl http://0.0.0.0:8000/v1/models
```

如果未运行，重新启动 vLLM 服务。

### Q: API 密钥报错

检查环境变量是否正确设置：

```bash
echo $DEEPSEEK_API_KEY
```

或在 `experiment_config.yaml` 中直接填写密钥（注意不要提交到 git）。

### Q: 如何对比两个不同模型？

分别用不同的 `model_name` 运行两次评测，然后同时生成表格：

```yaml
# 第一次运行
llm:
  model_name: Qwen3-8B

# 第二次运行（改为新模型）
llm:
  model_name: Qwen3-4B
```

生成表格时两个模型会并排显示：

```bash
python report/generate_dataset_tables.py
python report/generate_summary.py
```

### Q: 模型不支持 `response_format` 结构化输出怎么办？

框架会自动检测并降级处理：先尝试 `parse` 模式（原生结构化输出），失败则降级为 `create` 模式（在 Prompt 中注入 JSON 格式要求，然后用正则提取）。通常无需手动干预。

### Q: 思考（thinking）模式怎么控制？

由协议决定，不再手动配置：`direct` 关闭 thinking（裸答），`direct_think` / `cot` / `del_tom` 开启。关闭时请求会带 `extra_body: {enable_thinking: false, chat_template_kwargs: {enable_thinking: false}}`；开启时对 Qwen3 等模型触发 Chain-of-Thought，推理过程保存到 `prediction.jsonl` 的 `pred.reasoning` 字段。详见 [protocols.md](protocols.md)。

### Q: 随机抽样（max_samples > 0）是否可复现？

是的，`load_and_limit_data` 内部使用固定随机种子，相同 `max_samples` 每次抽取的样本集完全一致。
