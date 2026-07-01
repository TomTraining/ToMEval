# 智能体(Agent)评测接入方案

## 目标
在现有 ToMEval 框架里新增一条「智能体评测」路径:参赛方提交一个 HTTP 服务(内部代码自写、协议不限、可多轮/多次调模型),
我们统一提供本地 vLLM(OpenAI 兼容)作为唯一 model,用**现有 19 个数据集**评测其准确率,并**记录效率指标(不进排名)**。

judge / metric / 19 个数据集 / 报告主链路**一行不改**,只在「预测」这一层加黑盒 agent 后端。

## 提交与运行流程(已确认)
1. 参赛方在环境变量里用**占位符**表示模型连接信息,并按统一接口启动一个 HTTP 服务。
2. 我们收到后,把占位符替换成本地 vLLM 的真实 `api_url / api_key / model`。
3. 通过其 HTTP 服务逐条输入评测数据,得到预测。
4. 计算 metrics + 效率指标。
5. **同一时间只评测一个 agent(串行)** —— 这段时间打到 vLLM 的全部流量都归当前 agent,效率计量几乎白送。

```
参赛 agent(HTTP 服务) ──调用──► 本地 vLLM(OpenAI 兼容, 统一 model)
        ▲
        │我们发题(/predict) / 收答案
        ▼
   评测框架:build_option_bundle → AgentClient → pred.content → judge(老逻辑) → metrics
                                                            └► efficiency.json(vLLM /metrics 前后差值)
```

## 三个契约

### ① 启动契约(统一接口)
环境变量(agent 用占位符,我们替换):
- `LLM_API_URL` → vLLM base_url
- `LLM_API_KEY` → vLLM key
- `LLM_MODEL`   → 统一 model 名
- `PORT`        → 我们指定 agent 监听端口

HTTP 端点(agent 必须实现):
- `GET /health`  → 就绪返回 200(我们轮询到 200 才发题)
- `POST /predict` → 收一条样本,返回一条预测

### ② 预测契约(我们 → agent,每样本一条,绝不泄露答案)
```json
{
  "sample_id": "tomi_0012",
  "prompt_type": "mcq_single",     // mcq_single | mcq_multi | open | mcq_grouped
  "lang": "en",
  "story": "...",
  "question": "...",
  "options": {"A": "...", "B": "...", "C": "..."}   // open 题无此字段
}
```
- `options` 由我们跑 `build_option_bundle`(带 shuffle、种子只依赖 `dataset|sample_id|repeat`,可复现)生成。
- `correct_letters` 留在我们这边,**不发给 agent**。
- **关键约束**:MCQ 的字母是我们生成的,agent 必须用我们发的 `options` 里的字母作答,判分才对得上 —— 写进参赛文档。

### ③ 输出契约(agent → 我们,每样本一条)
```json
{"sample_id": "tomi_0012", "prediction": "A"}        // mcq_single: 字母
{"sample_id": "...",       "prediction": "A,C"}      // mcq_multi:  逗号分隔
{"sample_id": "...",       "prediction": ["A","B"]}  // mcq_grouped:每子问一个(二期)
{"sample_id": "...",       "prediction": "自由文本"}  // open
```
缺失 / 超时 / 非法 → `prediction=None` → 落进现有 `content_none` 桶判错,**绝不 crash**。

## 效率计量(只记录,不进排名)
- **主口径**:评测窗口前后各读一次 vLLM Prometheus `/metrics`,差值即当前 agent 的**全部**消耗(串行保证归属,agent 伪造不了)。
- 记录:`total_tokens`(prompt/completion 可分开)、`total_model_calls`、`wall_clock_seconds`,派生 `tokens_per_sample`/`calls_per_sample`。
- 落盘到 `efficiency.json`,与 `metrics.json` 同目录。排名方式后续再定。

## 代码改动清单(judge / 19 数据集 / 报告都不动)

1. **`src/llm/agent_client.py`(新增)** `AgentClient`
   - 签名对齐 `ContentClient.batch_generate(prompts, desc, system_prompts)`,让 `predict_records` 几乎不用改。
   - 内部:并发(复用 `max_workers`)打 agent `/predict`,超时 + 重试 + 失败降级(返回 `LLMResponse(content=None)`)。
   - 由于 `/predict` 需要结构化字段(不只是 rendered prompt),`predict_records` 传参需微调(见 4)。

2. **`src/evaluation/prompts.py`** `extract_prediction_from_text`
   - 加 `extractor == "agent"` 分支:直通归一化 `prediction`(字母大写、mcq_multi 拆逗号成列表、open 原样),不再找 `\boxed{}`。

3. **`src/evaluation/protocols.py`**
   - `extractor_name_for` 认 agent 模式返回 `"agent"`;agent 模式下 `repeats=1`、`shuffle` 沿用默认(True)。
   - 用一个独立配置项 `predictor: agent`(或 `protocol: agent`)标识,避免和现有 4 个 protocol 的采样表冲突。

4. **`src/evaluation/prediction.py`**
   - agent 模式下:仍复用前半段生成 `option_map / prompt_type / lang`;把结构化字段(sample_id/prompt_type/lang/story/question/options)交给 `AgentClient`,而非只传 rendered prompt。
   - 最小侵入方式:`AgentClient.batch_generate` 额外接一个 payloads 列表(通过 client 实例携带,或新增可选参数),保持 `predict_records` 主流程不变。

5. **`src/evaluation/pipeline.py`(预测阶段)**
   - agent 模式:预测前起 agent 进程(替换环境变量占位符)、轮询 `/health`;预测前后抓 vLLM `/metrics` 快照;评完关闭 agent,写 `efficiency.json`。
   - 封装到新增 `src/evaluation/agent_launcher.py`(起停 agent + `/metrics` 快照 + 差值)。

6. **`experiment_config.yaml`**
   - 新增 agent 段:`agent.start_command` / `agent.port` / `agent.health_timeout` / `agent.predict_timeout`,以及 vLLM `metrics_url`。
   - `predictor: agent`(或复用 `protocol` 字段值 `agent`)。

7. **report(可选,二期)**:效率表并入报告;一期先只落 `efficiency.json`。

## 分期
- **一期**:mcq_single / mcq_multi / open 三种题型 + 准确率 + 效率落盘。跑通单数据集(如 ToMi)验证。
- **二期**:mcq_grouped(EmoBench,走 `meta.sub_questions`);效率表进报告;排名合成方式。

## 验证
- 先写一个最小 mock agent(占位符 → vLLM,`/predict` 直接透传给 vLLM 单轮问答),
  在 `ToMi` 上跑 `stage=all`,确认:prediction.jsonl 有 agent 预测、metrics.json 正常、efficiency.json 有非零 token。
- 再验证一个「多次调模型 / 多轮」的 mock,确认 `/metrics` 差值随之上升。
- 失败注入:让 mock 对某些样本超时/返回非法,确认落进 `content_none` 判错、不 crash。
