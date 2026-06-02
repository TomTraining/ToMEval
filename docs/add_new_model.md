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
# 被测模型（LLM）配置
# ============================
llm:
  model_name: MyNewModel        # 模型名称，同时作为 results/ 下的目录名
  api_key: not-needed           # API 密钥（本地服务填 not-needed，云端 API 填真实密钥）
  api_url: http://0.0.0.0:8000/v1   # API 端点
  temperature: 0.6              # 采样温度
  max_tokens: 32768             # 最大输出 token 数
  max_workers: 64               # 并发线程数
  enable_thinking: false        # 是否启用思考模式（仅 Qwen3 等支持 thinking 的模型）
  system_prompt: ""             # 系统 Prompt（留空则不设置）

# ============================
# Judge 模型配置（用于判断答案正确性）
# ============================
judge:
  model_name: MyNewModel
  api_key: not-needed
  api_url: http://0.0.0.0:8000/v1
  temperature: 0.0              # Judge 建议用 0 以获得确定性输出
  max_tokens: 4096
  enable_thinking: false
  system_prompt: ""

# ============================
# 实验参数
# ============================
repeats: 3          # 重复运行次数（取平均以减少随机性）；使用云端 API 建议设为 1
max_samples: 0      # 0 = 全量；>0 = 随机抽取指定数量（用于快速测试）
normalized_datasets_path: datasets
results_path: results
```

### 重要说明

- `model_name` 的值会直接作为 `results/{dataset}/{model_name}/` 的目录名，**不同模型必须使用不同名称**。
- `api_key` 支持环境变量写法：`api_key: ${DEEPSEEK_API_KEY}`，运行时自动替换。
- `repeats: 3` 意味着每个数据集每条样本会被推理 3 次，最终指标取平均，适合本地推理；云端 API 计费时建议 `repeats: 1`。

---

## 步骤 2：运行评测

### 运行全部数据集

```bash
cd /path/to/ToMEval
python run_all.py
```

这会依次运行 `run_all.py` 中 `DATASETS` 列表里所有启用的数据集（当前 6 个：BigToM, EmoBench, FanToM, HiToM, SocialIQA, ToMBench）。

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
max_samples: 3   # 只取 3 条样本
repeats: 1       # 只跑 1 轮
```

```bash
python tasks/ToMBench/run.py
```

验证成功后再改回全量配置。

### 运行进度

运行时会实时打印进度：

```
已加载 1012 条样本，数据集: ToMBench/test
开始推理，共 3036 个请求（1012 样本 × 3 轮）...
第 1 轮: accuracy=0.7321  (741/1012)
第 2 轮: accuracy=0.7343  (743/1012)
第 3 轮: accuracy=0.7356  (744/1012)
平均 accuracy: 0.7340 ± 0.0018
结果已保存到: results/ToMBench/MyNewModel/exp_20260422_143022/
```

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

每行是一个 JSON 对象，记录单条样本的预测详情：

```json
{
  "repeat": 0,
  "sample_idx": 0,
  "gold_answer": "B",
  "pred": {
    "content": {"answer": "B"},
    "reasoning": "Let me think step by step..."
  },
  "prompt": "Read the following story...",
  "meta": {"ability": "Belief: Content false beliefs"},
  "is_correct": true,
  "error_reason": null
}
```

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
    --enable-reasoning \
    --reasoning-parser deepseek_r1

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
  enable_thinking: true         # 如启用了 reasoning parser
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
  temperature: 0.6
  max_tokens: 8192
  max_workers: 16      # 云端 API 建议降低并发
  enable_thinking: false
```

```bash
export DEEPSEEK_API_KEY="sk-xxx"
python run_all.py
```

#### OpenAI

```yaml
llm:
  model_name: gpt-4o
  api_key: ${OPENAI_API_KEY}
  api_url: https://api.openai.com/v1
  temperature: 0.6
  max_tokens: 4096
  max_workers: 8
  enable_thinking: false
```

```bash
export OPENAI_API_KEY="sk-xxx"
python run_all.py
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

### llm 配置

| 参数 | 说明 | 建议值 |
|---|---|---|
| `model_name` | 模型名称，也是 `results/` 下的目录名 | 使用清晰可辨的名称 |
| `api_url` | API 端点，末尾不要加 `/` | - |
| `api_key` | API 密钥，支持 `${ENV_VAR}` 变量替换 | - |
| `temperature` | 采样温度，0.0 确定性，1.0 最随机 | `0.6`（推理任务） |
| `max_tokens` | 最大输出 token 数 | `32768`（本地），`4096`（云端） |
| `max_workers` | 并发线程数 | `64`（本地），`8-16`（云端） |
| `enable_thinking` | 启用 Qwen3 等模型的思考模式 | 仅支持 thinking 的模型设 `true` |
| `system_prompt` | 系统提示词 | `""` 使用各数据集默认配置 |

### judge 配置

| 参数 | 说明 | 建议值 |
|---|---|---|
| `temperature` | Judge 温度 | `0.0`（确定性输出） |
| `max_tokens` | Judge 输出长度 | `4096` |

### 实验参数

| 参数 | 说明 | 建议值 |
|---|---|---|
| `repeats` | 重复轮数 | `3`（本地），`1`（云端 API） |
| `max_samples` | 最大样本数，`0` 为全量 | `0`（正式），`3-10`（调试） |
| `normalized_datasets_path` | 标准化数据集目录 | `datasets` |
| `results_path` | 结果输出目录 | `results` |

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

### Q: enable_thinking 有什么作用？

`enable_thinking: true` 会在请求中添加 `extra_body: {chat_template_kwargs: {enable_thinking: true}}`，触发 Qwen3 等模型的思考（Chain-of-Thought）推理，推理过程会保存到 `prediction.jsonl` 的 `pred.reasoning` 字段中。仅对支持 thinking 模式的模型有效。

### Q: 随机抽样（max_samples > 0）是否可复现？

是的，`load_and_limit_data` 内部使用固定随机种子，相同 `max_samples` 每次抽取的样本集完全一致。
