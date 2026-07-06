# Agent 模板

你**只需要实现一个函数** `predict`,就能完成接入。

```python
# solution.py
def predict(sample: dict, model: dict) -> str | list:
    ...
```

## 目录说明

| 文件 | 你要改吗 | 作用 |
|---|---|---|
| `solution.py` | ✅ **只改这里** | 实现 `predict(sample, model)`;已附带一个可直接跑的单轮 baseline |
| `selftest.py` | ❌ 不用改 | 本地自测:用四种题型样本调用你的 `predict`,校验返回格式 |
| `mock_model_server.py` | ❌ 不用改 | 自测用的模拟模型(假 OpenAI 接口,返回随机合法答案) |
| `mock_samples.json` | ❌ 不用改 | 自测用的四种题型示例数据 |

你可以在仓库里自由新增自己的模块 / 依赖,`solution.py` 里正常 `import` 即可。

## 输入

`predict` 每次收到一道题目 `sample` 和一份模型连接信息 `model`。

**`sample`**(不含标准答案):

| 字段 | 说明 |
|---|---|
| `sample_id` | 样本 id |
| `prompt_type` | 题型:`mcq_single` / `mcq_multi` / `mcq_grouped` / `open` |
| `lang` | `"en"` 或 `"zh"` |
| `story` | 故事/背景(少数题型可能为空串) |
| `question` | 问题 |
| `options` | 选项字母→文本,如 `{"A": "...", "B": "..."}`;`open` 题无此字段 |
| `sub_questions` | 仅 `mcq_grouped`:每项含自身 `question` 与 `options`,按顺序作答 |

**`model`**(调用模型的连接信息,每条请求下发,**请勿写死**):

| 字段 | 说明 |
|---|---|
| `api_url` | OpenAI 兼容 base_url |
| `api_key` | 调用用的 key |
| `model_name` | 模型名 |

直接用它构造 OpenAI 客户端即可:

```python
from openai import OpenAI
client = OpenAI(api_key=model["api_key"], base_url=model["api_url"])
resp = client.chat.completions.create(
    model=model["model_name"],
    messages=[{"role": "user", "content": "..."}],
)
```

## 输出(严格格式)

`predict` 的返回值必须严格符合题型格式,否则该题判错(不会做二次提取/纠正):

| `prompt_type` | 返回值 | 示例 |
|---|---|---|
| `mcq_single` | 单个大写字母字符串 | `"A"` |
| `mcq_multi` | 大写字母数组,升序、去重、至少一个 | `["A", "C"]` |
| `mcq_grouped` | 大写字母数组,每子问一个、顺序对应 `sub_questions` | `["A", "B"]` |
| `open` | 非空文本字符串 | `"He thinks ..."` |

**MCQ 必须使用 `options` 里给定的字母作答**(选项顺序已被打乱,返回选项文本或自编编号会判错)。
`mcq_multi` / `mcq_grouped` 须返回**字母数组**,不是逗号分隔的字符串 `"A,C"`。

## 本地自测

提交前先跑自测,确认你的 `predict` 返回格式合规、链路能跑通:

```bash
# 默认:用内置模拟模型,无需真实模型/网络,最省事
python selftest.py

# 或:用你自己的模型(如本地 vLLM)
MODEL_API_URL=http://127.0.0.1:8000/v1 MODEL_API_KEY=x MODEL_NAME=your-model python selftest.py
```

自测会对四种题型逐条打印 `PASS` / `FAIL`。全部 `PASS` 表示格式合规、链路通(注意:模拟模型答案是随机的,`PASS` 不代表答对)。

## 提交

提交整个仓库,根目录必须包含 `solution.py`,其中定义 `predict(sample, model)` 函数。把你用到的依赖一并写进仓库(如 `requirements.txt`)。
