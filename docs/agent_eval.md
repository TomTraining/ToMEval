# 智能体(Agent)评测对接指南

本文定义 ToMEval 对**智能体(Agent)**的评测接口:参赛方需交付的 HTTP 服务契约、输入输出格式、错误处理与启动方式。

- 设计动机与内部实现:见 [agent_eval_plan.md](agent_eval_plan.md)
- 数据集说明:见 [datasets.md](datasets.md)
- 普通模型(非 Agent)评测的四个协议:见 [protocols.md](protocols.md)

术语:下文以「框架」指代 ToMEval 评测框架(负责出题与判分),以「Agent」指代参赛方交付的 HTTP 服务(负责答题)。

---

## 1. 概述

Agent 是一个黑盒 HTTP 服务,实现 `GET /health` 与 `POST /predict` 两个端点。框架逐条下发结构化题目,Agent 返回答案,该答案直接作为预测进入判分。Agent 内部如何构造 prompt、调用哪个模型、单轮还是多轮、是否使用工具,均不受约束。

### 职责划分

| 职责 | 承担方 |
|---|---|
| 数据集准备、抽样、生成并打乱选项 | 框架 |
| 逐条下发结构化题目 | 框架 |
| prompt 构造、模型选择、单/多轮、工具使用 | Agent |
| 从模型输出中提取最终答案 | Agent |
| 判分、准确率、效率统计 | 框架 |

核心原则:框架仅下发结构化输入(§5)并规定每种题型的输出格式(§6);Agent 返回的 `prediction` 字段**原样进入判分,框架不做任何二次提取或改写**。因此「从模型自由文本中提取出符合格式的答案」由 Agent 负责。

---

## 2. 参考实现

仓库提供两个可直接运行的参考 Agent(Python 标准库 `http.server`),可作为脚手架:

| 路径 | 策略 |
|---|---|
| `agents/mock/server.py` | 单轮:每题调用一次模型 |
| `agents/mock_multicall/server.py` | 多轮:每题调用两次模型(先分析、再定答案) |

本地手动试跑一个 Agent(正式评测时由框架自动启动,见 §4):

```bash
LLM_API_URL=<模型地址> LLM_API_KEY=<key> LLM_MODEL=<模型名> PORT=8100 \
  python agents/mock/server.py

# 另开一个终端验证:
curl -s localhost:8100/health
curl -s localhost:8100/predict -d '{"sample_id":"t1","prompt_type":"mcq_single",
  "lang":"en","story":"...","question":"...","options":{"A":"foo","B":"bar"}}'
```

---

## 3. 单条样本的流转

```
① 框架从数据集取一条样本(story + question + 标准答案),
   本地生成选项字母并打乱顺序:A. treasure_chest  B. crate
   标准答案(A)不下发。
        │
        ▼
② 框架 POST 一条结构化题目至 Agent 的 /predict(不含答案):
   {sample_id, prompt_type:"mcq_single", lang, story, question,
    options:{A:"treasure_chest", B:"crate"}}
        │
        ▼
③ Agent 内部自行处理(构造 prompt / 单轮或多轮 / 调用模型 / 使用工具),
   并自行从模型输出中提取最终答案。
        │
        ▼
④ Agent 返回预测:{sample_id, prediction:"A"}   —— 原样进入判分
        │
        ▼
⑤ 框架判分:prediction "A" 与标准答案 "A" 比对 → 正确
        │
        ▼
⑥ 全部样本判完 → 计算准确率,统计 token 消耗与模型调用次数
```

要点:

- 逐条下发,Agent 不接触整个数据集,也不接触任何标准答案。
- 判分由框架完成,准确率不可自报。
- 选项字母由框架生成并打乱;MCQ 题必须使用 `options` 中给定的字母作答,否则判分无法对齐(见 §6)。

---

## 4. 模型访问与启动

### 启动方式

参赛方交付的 Agent 目录中**必须包含一个名为 `server.py` 的启动入口**。框架在 Agent 目录下以固定命令启动服务:

```
python server.py
```

`server.py` 须从环境变量读取连接信息与端口,并在该端口启动 HTTP 服务:

| 环境变量 | 含义 |
|---|---|
| `PORT` | HTTP 服务须监听的端口 |
| `LLM_API_URL` | 模型的 OpenAI 兼容地址 |
| `LLM_API_KEY` | 调用模型使用的 key |
| `LLM_MODEL` | 统一模型名(所有参赛方相同) |

以上均不得写死,须从环境变量读取。

### 模型调用

`LLM_API_URL` 指向框架的一层代理,而非模型本体;使用标准 OpenAI 客户端照常调用即可,后端形态(本地推理或外部 API)对 Agent 透明。代理会统计 token 消耗与模型调用次数用于效率记录(响应中的 `usage` 字段不回传)。效率数据仅作记录、不计入排名,请合理使用模型。

调用示例(Python):

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ["LLM_API_URL"],
)

resp = client.chat.completions.create(
    model=os.environ["LLM_MODEL"],
    messages=[{"role": "user", "content": "..."}],
    temperature=0.0,
)
answer = resp.choices[0].message.content
```

### HTTP 端点

#### `GET /health`

服务就绪后返回 HTTP 200。框架在下发题目前轮询该端点,直到返回 200 才开始评测(默认最长等待 120 秒)。启动耗时(加载模型、建索引等)期间应保持非 200,以避免过早收到请求。

```
GET /health   →   200   {"status": "ok"}
```

#### `POST /predict`

接收一条样本(§5),返回一条预测(§6)。框架会并发调用该端点,因此其处理须**可并发、无状态**——样本之间不得共享可变状态。单条请求默认超时 300 秒。

---

## 5. 请求格式(框架 → Agent)

框架 POST 至 `/predict` 的请求体为一条结构化样本。完整 JSON Schema 见 [`agent_schema/predict_request.schema.json`](agent_schema/predict_request.schema.json)。

```json
{
  "sample_id": "tomi_0012",
  "prompt_type": "mcq_single",
  "lang": "en",
  "story": "James entered the attic. Ava ...",
  "question": "Where does James think that Ava searches for the pants?",
  "options": {"A": "treasure_chest", "B": "crate"}
}
```

| 字段 | 必有 | 说明 |
|---|---|---|
| `sample_id` | ✓ | 样本唯一 id,须在响应中原样回填 |
| `prompt_type` | ✓ | 题型:`mcq_single` / `mcq_multi` / `mcq_grouped` / `open`,决定输出格式(见 §6) |
| `lang` | ✓ | `en` 或 `zh` |
| `story` | ✓ | 故事/背景正文(少数题型可能为空串) |
| `question` | ✓ | 问题 |
| `options` | 视题型 | 选项字母→文本映射。`mcq_single` / `mcq_multi` 必有;`open` 无此字段;`mcq_grouped` 见 `sub_questions` |
| `sub_questions` | 仅 grouped | `mcq_grouped` 专用:同一故事下的多道子问,每项含自身的 `question` 与 `options`,按顺序作答 |

`mcq_grouped` 的请求形态(一个故事,多道子问):

```json
{
  "sample_id": "s1", "prompt_type": "mcq_grouped",
  "lang": "en", "story": "...", "question": "...",
  "sub_questions": [
    {"question": "Q1 ...", "options": {"A": "...", "B": "..."}},
    {"question": "Q2 ...", "options": {"A": "...", "B": "..."}}
  ]
}
```

---

## 6. 响应格式(Agent → 框架)

成功时返回携带 `prediction` 的对象;失败时返回携带 `error` 的对象;二者恰有其一。完整 JSON Schema 见 [`agent_schema/predict_response.schema.json`](agent_schema/predict_response.schema.json)。

### 成功

```json
{"sample_id": "tomi_0012", "prediction": "A"}
```

`prediction` 的格式取决于题型。Agent 须在内部将模型输出整理为下列标准格式后返回,框架据此原样判分:

| `prompt_type` | `prediction` 格式 | 示例 |
|---|---|---|
| `mcq_single` | 单个大写字母(字符串) | `"A"` |
| `mcq_multi` | 大写字母数组,升序、去重、至少一个 | `["A", "C"]` |
| `mcq_grouped` | 大写字母数组,每个子问一个、顺序对应 `sub_questions`,长度等于子问数(允许重复) | `["A", "B"]` |
| `open` | 非空自由文本(字符串) | `"He thinks she will look in the crate."` |

两处约束:

1. MCQ 题必须使用请求 `options` 中给定的字母(`A`/`B`/`C`...)。选项顺序经随机打乱,返回选项文本或自编编号将判错。
2. `mcq_multi` / `mcq_grouped` 须返回**字母数组**(`["A","C"]`),不是逗号分隔的字符串 `"A,C"`。

### 失败

模型调用失败、超时、题型不支持等情形,返回携带 `error` 的对象(而非 `prediction`):

```json
{"sample_id": "tomi_0012",
 "error": {"code": "MODEL_ERROR", "retryable": true, "message": "upstream 502"}}
```

| 字段 | 说明 |
|---|---|
| `code` | 标准错误码,见 §7 |
| `retryable` | `true` 时框架退避后重试该样本;`false` 时直接判该样本为错。须与 `code` 语义一致 |
| `message` | 可选,人读的错误描述,截断至 200 字 |

---

## 7. 错误码与 HTTP 状态

失败时以对应 HTTP 状态码返回 §6 的 error 对象。标准错误码:

| `code` | HTTP | `retryable` | 含义 |
|---|---|---|---|
| `MODEL_TIMEOUT` | 504 | true | 模型调用超时 |
| `MODEL_ERROR` | 502 | true | 模型调用返回错误(可能瞬时) |
| `OVERLOADED` | 503 | true | 服务过载,稍后重试 |
| `UNSUPPORTED_PROMPT_TYPE` | 422 | false | 收到不支持的 `prompt_type` |
| `INVALID_REQUEST` | 400 | false | 请求体非法(坏 JSON、缺字段) |
| `INTERNAL` | 500 | false | 其它内部错误 |

Agent 返回的任何失败都不会中断评测:框架将该样本记为答错(归入 `content_none`)。因此内部异常应捕获并返回 error 对象(或 `prediction: null`),不得使服务崩溃或挂起。当无法得到有效答案时,返回一个合法格式的答案(即使是随机选项)与返回 `content_none` 在判分上等价,但前者不会因服务异常影响后续并发请求。

---

## 8. 提交前自检清单

- [ ] Agent 目录包含 `server.py` 启动入口,`python server.py` 可启动服务。
- [ ] 服务监听 `PORT` 指定的端口(不写死)。
- [ ] 从 `LLM_API_URL` / `LLM_API_KEY` / `LLM_MODEL` 读取模型连接信息(不写死)。
- [ ] `GET /health` 就绪后返回 200,未就绪时返回非 200。
- [ ] `POST /predict` 可并发、样本间无状态。
- [ ] 输出符合 §6:MCQ 使用给定字母;`mcq_multi` / `mcq_grouped` 返回字母数组而非字符串;`open` 返回非空文本。
- [ ] 响应中 `sample_id` 原样回填。
- [ ] 内部异常有兜底,返回 error 对象或 `prediction: null`,服务不崩溃、不挂起。
