# data_eval — 合成数据质量评估模块

对 `feedback_data/synth_clean/<dataset>/synthetic_iterN_<model>.parquet` 这类合成训练数据，
从 4 个维度做质量评估，并汇总为一份 markdown 报告。

## 评估维度

| 维度 | 含义 | 实现 | 是否调 LLM |
| --- | --- | --- | --- |
| `format` | 字段/类型/取值合规性 | `eval_format.py` + `format_rules.py`（规则来源 `_audit/<DS>.json`） | 否 |
| `difficulty` | 1-5 难度分（强模型打分 + 简模型 repeat=5 通过率） | `eval_difficulty.py` | 是 |
| `answerability` | 前提自洽 / 答案唯一 / 逻辑一致（A→B→C 三阶段） | `eval_answerability.py` + `answerability_core.py` | 是 |
| `representativeness` | 维度代表性 0-5 分 + 维度细分 | `eval_representativeness.py` | 是 |

`report.py` 不参与打分，只用 `report_model` 把上面 4 项结果汇总成自然语言报告。

## 目录结构

```
data_eval/
├── README.md                       本文件
├── __init__.py
├── config.yaml                     运行配置（API key、采样、路径、运行参数）
├── base.py                         配置加载 / parquet 加载 / 报告写出 / 公共 dataclass
├── format_rules.py                 从 _audit JSON 推导的格式规则（F034）
├── eval_format.py                  format 评估
├── eval_difficulty.py              difficulty 评估
├── eval_answerability.py           answerability 评估
├── answerability_core.py           answerability 阶段 A/B/C 共用核心
├── eval_representativeness.py      representativeness 评估
├── report.py                       markdown 报告生成
└── _audit/                         8 份数据集结构快照（format 规则来源）
    ├── BigToM.json
    ├── EmoBench.json
    ├── FanToM.json
    ├── HiToM.json
    ├── SimpleToM.json
    ├── SocialIQA.json
    ├── ToMBench.json
    └── summary.json
```

入口脚本 `run_eval.py` 在仓库根目录（不在本目录）。

## 配置 `config.yaml`

发布版本中所有 `api_key` 都已替换为占位符 `<YOUR_API_KEY>`，**使用前必须先填入真实 key**，否则调用 LLM 时会被 API 直接拒绝。

```yaml
eval_model:
  strong:                                  # 难度打分 / answerability B,C / representativeness 打分
    api_url: https://dashscope.aliyuncs.com/compatible-mode/v1
    api_key: <YOUR_API_KEY>                # ← 替换为真实 key
    model_name: deepseek-v4-flash
    max_workers: 8
    temperature: 0.6
    max_tokens: 8192
  simple:                                  # answerability 阶段 A 复用其在 difficulty 阶段的 repeat=5 产物
    api_url: https://dashscope.aliyuncs.com/compatible-mode/v1
    api_key: <YOUR_API_KEY>                # ← 替换为真实 key
    model_name: qwen3-8b
    ...

report_model:                              # 仅做自然语言归纳，不打分
  api_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  api_key: <YOUR_API_KEY>                  # ← 替换为真实 key
  model_name: deepseek-v4-flash
  ...

datasets: [BigToM, EmoBench, FanToM, HiToM, SocialIQA, ToMBench]

sample_rows:                               # LLM 评估每子集采样上限；format 走全行不受此控制
  default: 50
  per_dataset: {}                          # 可按数据集覆盖，如 BigToM: 30

paths:
  train_root: train_datasets               # 评估数据所在根（按需修改）
  output_root: train_eval_output           # 中间产物 / 子集 JSON
  audit_root: data_eval/_audit             # format 规则来源
  report_path: docs/train_data_quality_report.md   # 最终 markdown 报告路径

run:
  datasets: null                           # null = 用顶层 datasets 全量；非空时必须 ⊆ 顶层 datasets
  report_only: false                       # true = 跳过评估，仅基于已有 JSON 重渲 markdown
  no_report: false                         # true = 跑评估但不生成 markdown
```

> **不要把填好真实 key 的 `config.yaml` 提交回公开仓库。** 推荐做法：本地复制一份不入版本管理（`.gitignore` 加规则），或仅在私有部署环境写真值。

## 用法

入口在仓库根目录的 `run_eval.py`，按维度分别执行或一把跑全量：

```bash
# 单维度
python run_eval.py --eval format             --dataset BigToM
python run_eval.py --eval difficulty         --dataset BigToM --max-rows 10
python run_eval.py --eval answerability      --dataset FanToM --iter 1 --model "*"
python run_eval.py --eval representativeness --dataset FanToM

# 一把跑四个维度
python run_eval.py --eval all --dataset FanToM --iter 1 --model "*"
```

CLI 参数：

| 参数 | 含义 | 默认 |
| --- | --- | --- |
| `--eval` | `format` / `difficulty` / `answerability` / `representativeness` / `all` | 必填 |
| `--dataset` | 数据集名，如 `BigToM` | 必填 |
| `--iter` | 迭代轮次 | `1` |
| `--model` | 模型名（支持 glob，如 `*`） | `*` |
| `--max-rows` | 限制处理行数（用于快速冒烟） | 不限 |
| `--root` | synth_clean 根目录 | `feedback_data/synth_clean` |
| `--output-root` | 评估报告输出根目录 | `data_eval_output` |

> CLI 参数和 `config.yaml.paths/run` 的关系：`run_eval.py` 当前以 CLI 参数为准；`config.yaml.paths/run` 段是为后续"零 CLI 全配置驱动"留的入口，不影响现有命令。

## 输入数据约定

`load_synth_parquet(dataset, iter_n, model, root)` 在
`<root>/<dataset>/synthetic_iter<N>_<model>.parquet` 下查找文件，自动排除 `_hard.parquet`。
`model` 支持 glob，命中多个时取第一个（按文件名排序）。

## 输出

- 每个维度的子集结果 JSON：`<output-root>/<eval>/<dataset>_<file_stem>.json`
- 汇总 markdown 报告：通过 `report.py:generate_quality_report(results, out_path)` 生成，路径由 `config.yaml.paths.report_path` 指定（默认 `docs/train_data_quality_report.md`）。

## 退出码（`run_eval.py`）

- `0` — 正常完成
- `1` — `format` 评估有失败行（其他维度不通过 pass/fail，只输出统计）
- `2` — 找不到对应的输入数据文件

## 依赖

- 数据加载：`pandas` + `pyarrow`（读 parquet）
- 配置：`PyYAML`
- LLM 客户端：仓库内 `src/llm/content_client.ContentClient`（OpenAI 兼容协议）

## 注意事项

- `format_rules.load_rules(dataset)` 会从 `data_eval/_audit/<DS>.json` 读规则，**缺文件直接抛错**（不回落到硬编码）。本分支已包含 7 个数据集 + `summary.json` 的快照。
- `eval_model.strong` / `eval_model.simple` 必须都配齐，缺角色 → `ValueError`。
- `sample_rows.default` 必须为正整数；`per_dataset` 可缺省。
- `run.report_only` 与 `run.no_report` 不能同时为 `true`。
