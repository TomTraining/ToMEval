# 智能体(Agent)评测对接指南

本文定义 ToMEval 对**智能体(Agent)**的评测接口:参赛方需交付的 HTTP 服务契约、输入输出格式、错误处理与启动方式。

- 设计动机与内部实现:见 [agent_eval_plan.md](agent_eval_plan.md)
- 数据集说明:见 [datasets.md](datasets.md)
- 普通模型(非 Agent)评测的四个协议:见 [protocols.md](protocols.md)

术语:下文以「框架」指代 ToMEval 评测框架(负责出题与判分),以「Agent」指代参赛方交付的 HTTP 服务(负责答题)。

---

## 1. 概述

参赛方交付一个**代码仓库**,根目录含 `solution.py`,其中实现 `predict(sample, model)`。评测时,框架把该仓库拉到 `agents/` 下,在本地起统一运行时 `agents/server.py` 动态加载其 `solution.py`,再逐条下发结构化题目、收回答案进入判分。`predict` 内部如何构造 prompt、单轮还是多轮、是否使用工具,均不受约束;调用模型用的连接信息由框架随每条请求下发(见 §4)。

参赛方**只写 `predict` 一个函数**,不接触 HTTP、并发、错误处理——那些都在框架自带的 `agents/server.py` 里(参赛方看不到也不需要它)。

### 职责划分

| 职责 | 承担方 |
|---|---|
| 数据集准备、抽样、生成并打乱选项 | 框架 |
| 起 `agents/server.py`、拆分请求体、逐条下发题目、并发、错误兜底 | 框架(`server.py`) |
| `predict`:构造 prompt、调模型、单/多轮、工具使用、提取最终答案 | 参赛方(`solution.py`) |
| 判分、准确率、墙钟/调用数记录 | 框架 |

核心原则:框架仅下发结构化输入(§5)并规定每种题型的输出格式(§6);`predict` 返回的 `prediction` **原样进入判分,框架不做任何二次提取或改写**。因此「从模型自由文本中提取出符合格式的答案」由参赛方在 `predict` 里负责。

---

## 2. 组成与参考实现

### 框架侧(我方,参赛方不可见)

| 路径 | 作用 |
|---|---|
| `agents/server.py` | 统一运行时:实现 `POST /predict`、`GET /health`,拆分请求体(`model` 与题目字段)、并发、异常→错误信封;从 `SOLUTION_DIR` 动态加载参赛方 `solution.py` 的 `predict` |
| `agents/mock/server.py` | 回归/示例:单轮 agent(单文件自实现整个 HTTP 服务) |
| `agents/mock_multicall/server.py` | 回归/示例:多轮 agent(每题调两次模型) |

### 参赛方交付物

参赛方拿到的是 `agents/template/`(一个**去框架化**的脚手架,不含 `server.py`、不暴露任何评测内部信息)。他们只需:

```python
# solution.py
def predict(sample: dict, model: dict) -> str | list:
    ...
```

`solution.py` 自带开箱即用的**单轮 baseline**,并附 `selftest.py` + 模拟模型供本地自测。参赛方交付整个仓库(根目录含 `solution.py`)。详见 `agents/template/README.md`(那份文档面向参赛方,不含 ToMEval 字样)。

#### 环境声明(可选:交 Dockerfile / requirements.txt)

为免「我本地能跑、评测机跑不起来」,参赛方可随仓库声明运行环境,框架用 Docker 起隔离容器评测(见 §4「运行时」):

- **只有 pip 依赖**:交一个 `requirements.txt` 即可,框架用默认基础镜像(`python:3.11-slim`)兜底安装。
- **需要系统包/特殊环境**:交一个 `Dockerfile`,自己把环境装好,把仓库拷进 `/agent`。

无论哪种,参赛方**都不要写 `ENTRYPOINT`/`CMD`、也不要在镜像里起任何 server**:HTTP 入口(`server.py`)由框架在构建时叠加进镜像并接管。参赛方职责始终只是 `solution.py` 里的 `predict`。模板自带一份可直接用的 `Dockerfile` + `requirements.txt`。

### 本地试跑(我方)

把参赛方仓库(或 `agents/template/` 本身)当作 `SOLUTION_DIR`,起统一运行时:

```bash
SOLUTION_DIR=agents/template PORT=8100 python agents/server.py

# 另开一个终端验证(model 凭证随请求下发):
curl -s localhost:8100/predict -d '{"sample_id":"t1","prompt_type":"mcq_single",
  "lang":"en","story":"...","question":"...","options":{"A":"foo","B":"bar"},
  "model":{"api_url":"https://.../v1","api_key":"sk-...","model_name":"qwen3-8b"}}'
```

实际评测中,上述 `SOLUTION_DIR` 与端口由 `experiment_config_agent.yaml` 的 `agent.solution_dir` / `agent.api_url` 指定,框架自动拉起、探活、评完关闭(见 §4)。

---

## 3. 单条样本的流转

```
① 框架从数据集取一条样本(story + question + 标准答案),
   本地生成选项字母并打乱顺序:A. treasure_chest  B. crate
   标准答案(A)不下发。
        │
        ▼
② 框架 POST 一条结构化题目至 Agent 的 /predict(不含答案,带 model 凭证):
   {sample_id, prompt_type:"mcq_single", lang, story, question,
    options:{A:"treasure_chest", B:"crate"},
    model:{api_url, api_key, model_name}}
        │
        ▼
③ Agent 内部自行处理(构造 prompt / 单轮或多轮 / 用 model 凭证调模型 / 使用工具),
   并自行从模型输出中提取最终答案。
        │
        ▼
④ Agent 返回预测:{sample_id, prediction:"A"}   —— 原样进入判分
        │
        ▼
⑤ 框架判分:prediction "A" 与标准答案 "A" 比对 → 正确
        │
        ▼
⑥ 全部样本判完 → 计算准确率,记录墙钟与调用次数
```

要点:

- 逐条下发,Agent 不接触整个数据集,也不接触任何标准答案。
- 判分由框架完成,准确率不可自报。
- 选项字母由框架生成并打乱;MCQ 题必须使用 `options` 中给定的字母作答,否则判分无法对齐(见 §6)。

---

## 4. 模型访问与端点

### 启动与端点

框架用统一运行时 `agents/server.py` 加载参赛方 `solution.py`,在本地起 HTTP 服务(`POST /predict`、`GET /health`),评完关掉。**参赛方无论选哪种运行时,交付物与契约都一样**(只写 `predict`);区别仅在框架怎么把 `server.py` 跑起来,由 `experiment_config_agent.yaml` 的 `agent.runtime` 选择:

- `runtime: local`(默认):框架 subprocess 直接起 `python agents/server.py`,`solution.py` 的依赖跑在评测机本机环境。轻量,适合我方本地试跑。
- `runtime: docker`:框架把参赛方仓库连同环境构建成镜像,再叠加 `server.py` 当 ENTRYPOINT 跑容器。环境彻底由参赛方决定(交 `Dockerfile` 或 `requirements.txt`),契约/并发/错误兜底仍是框架的 `server.py`。详见 §2「可选:交 Dockerfile 声明环境」。

两种运行时共用的配置项:

- `agent.solution_dir`:参赛方仓库根目录(含 `solution.py`)。local 模式注入为 `SOLUTION_DIR`;docker 模式作为镜像构建上下文(其 `Dockerfile`/`requirements.txt` 决定环境)。
- `agent.api_url`:本地服务监听地址,端口从中解析出来(local 注入为 `PORT`;docker 映射到容器的 8100)。
- `agent.health_timeout`:等 `/health` 就绪的最长秒数。

仅 `runtime: docker` 生效的配置项:`build_timeout`(单次 build 超时)、`docker_memory` / `docker_cpus` / `docker_pids_limit`(容器资源/进程数上限,防单个镜像拖垮评测机)、`docker_run_as_root`(默认非 root)。

框架 POST 到 `api_url` + `predict_path`(默认 `/predict`),等 `/health` 返回 200 后才发题。

> 防滥用说明:model 凭证直连我方后端、框架不中转,因此**框架侧看不到 agent 调了几次 model**,不做按调用数/token 的硬配额。防滥用靠单样本 `predict_timeout` + docker 模式的容器资源上限;真要限调用数须在 model 后端侧限流。
>
> 数据隔离说明:model 部署在评测机内网,评测服务器整体禁止访问外网。题目由框架主动 POST 进容器(入站),容器唯一的出站目标就是内网 model —— 参赛方代码即便想把题目外传到公网也无路由可走,隐藏测试集**天然不会外泄**,无需在 docker 层另配出口白名单。

### 模型调用

调用模型的连接信息**随每条 `/predict` 请求体的 `model` 字段下发**(不再走环境变量,也没有代理):

```json
"model": {"api_url": "https://.../v1", "api_key": "sk-...", "model_name": "qwen3-8b"}
```

该后端是我们部署的**统一 model**(所有参赛方相同),用标准 OpenAI 客户端照常调用即可。效率仅作记录、不计入排名,请合理使用模型。

调用示例(Python):

```python
from openai import OpenAI

def call_model(sample):
    m = sample["model"]
    client = OpenAI(api_key=m["api_key"], base_url=m["api_url"])
    resp = client.chat.completions.create(
        model=m["model_name"],
        messages=[{"role": "user", "content": "..."}],
        temperature=0.0,
    )
    return resp.choices[0].message.content
```

### `POST /predict`

接收一条样本(§5),返回一条预测(§6)。框架会并发调用该端点,因此其处理须**可并发、无状态**——样本之间不得共享可变状态。单条样本(含重试)默认总超时 300 秒。

> 上述 HTTP 端点、请求体拆分(把 `model` 与题目字段分开)、并发、异常兜底与错误信封均由框架侧的 `agents/server.py` 处理好,参赛方只需在 `solution.py` 里实现 `predict(sample, model)`,`sample` 为题目字段、`model` 为 `{api_url, api_key, model_name}`。

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
  "options": {"A": "treasure_chest", "B": "crate"},
  "model": {"api_url": "https://.../v1", "api_key": "sk-...", "model_name": "qwen3-8b"}
}
```

| 字段 | 必有 | 说明 |
|---|---|---|
| `sample_id` | ✓ | 样本唯一 id,须在响应中原样回填 |
| `prompt_type` | ✓ | 题型:`mcq_single` / `mcq_multi` / `mcq_grouped` / `open`,决定输出格式(见 §6) |
| `lang` | ✓ | `en` 或 `zh` |
| `story` | ✓ | 故事/背景正文(少数题型可能为空串) |
| `question` | ✓ | 问题 |
| `model` | ✓ | 调用模型的连接信息 `{api_url, api_key, model_name}`,我们部署的统一后端(见 §4) |
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

参赛方交付物是一个仓库,根目录含 `solution.py`(定义 `predict`)。以 `agents/template/` 为起点时:

- [ ] 在 `solution.py` 实现 `predict(sample, model)`,返回值严格符合 §6 的题型格式。
- [ ] 用传入的 `model`(`api_url` / `api_key` / `model_name`)调模型,不写死。
- [ ] `python selftest.py` 四种题型全 PASS(格式合规、链路通)。
- [ ] 依赖已写进仓库(如 `requirements.txt`)。
- [ ] 输出符合 §6:MCQ 使用给定字母;`mcq_multi` / `mcq_grouped` 返回字母数组而非字符串;`open` 返回非空文本。

评测侧(我方)自检:

- [ ] 参赛方仓库已拉到 `agents/<仓库名>/`,`agent.solution_dir` 指向它。
- [ ] `agent.api_url` 端口未被占用;`agents/server.py` 能加载到 `solution.py` 的 `predict`。
