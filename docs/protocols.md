# 评测协议（Protocol）说明

ToMEval 用「协议」统一控制一次评测**怎么问、怎么采样、怎么提取答案、要不要投票**。
在 `experiment_config.yaml` 里写一行 `protocol: <名称>` 即可，一次跑一个协议。

协议的全部逻辑集中在 `src/evaluation/protocols.py`（采样参数 / system prompt / extractor）
和 `src/evaluation/voting.py`（del_tom 投票），是单一事实来源。

---

## 一个协议决定四件事

| 维度 | 说明 |
|---|---|
| 采样参数 | `temperature` / `top_p` / `max_tokens` / `enable_thinking`，覆盖到 `llm` 配置上 |
| system prompt | 答题风格 + 答题格式指令（按「协议风格 × 题型 × 语言」生成），user prompt 只留故事+问题+选项 |
| extractor | 从模型输出里提取答案的方式（取第一个还是最后一个 `\boxed{}`） |
| 重复与投票 | 重复次数 = 协议的 `n_samples`；`del_tom` 跑 8 次后做多数投票 |

> `llm` 段只保留连接参数（model_name / api_key / api_url / max_workers）。
> 不设 `protocol` 时回退旧行为（采样参数走 `LLMClient` 内置默认、答题指令在 user prompt、extractor 取最后一个 `\boxed{}`）。

---

## 四个协议

| 协议 | temperature | top_p | max_tokens | enable_thinking | n_samples | extractor | 目的 |
|---|---|---|---|---|---|---|---|
| `direct` | 0.0 | 0.95 | 32768 | **false** | 1 | extract_direct | 测「裸答」能力（不思考） |
| `direct_think` | 0.6 | 0.95 | 32768 | true | 1 | extract_cot | 与历史评测口径对齐（默认思考） |
| `cot` | 0.6 | 0.95 | 32768 | true | 1 | extract_cot | 测链式推理（chain-of-thought） |
| `del_tom` | 0.6 | 0.95 | 32768 | true | **8** | extract_cot + 多数投票 | 多次采样投票，提升鲁棒性 |

说明：
- `max_tokens=32768` 是给本地 vLLM 的；部分云端 API（如 dashscope qwen3-8b）上限是 8192，需自行注意端点限制。
- `enable_thinking=false`（仅 direct）时会在请求里带 `extra_body: {enable_thinking: false, chat_template_kwargs: {enable_thinking: false}}`。
- 重复次数由 `n_samples` 派生，**忽略 config 里的 repeats**：direct/direct_think/cot 跑 1 次，del_tom 跑 8 次。

---

## system prompt

协议模式下（`experiment_config.yaml` 设了 `protocol`），答题格式指令从 user prompt **移到 system prompt**，
user prompt 只留故事/问题/选项。system prompt 按「风格 × 题型 × 语言」生成：

- **风格**两类：
  - **裸答**（`direct` / `direct_think`，`reasoning=False`）：直接给出最终答案（不硬性禁止解释，`direct_think` 开了 thinking）。
  - **推理**（`cot` / `del_tom`，`reasoning=True`）：一步步推理角色心理状态，最终答案放最后。
- **题型**四类：`mcq_single`（单选）/ `mcq_multi`（多选）/ `mcq_grouped`（一个 prompt 多子问，`prepare_samples` 打包）/ `open`（开放）。
- **语言**两套：英文 `en` / 中文 `zh`（跟随样本 `meta.lang` / `meta.language`）。

### system prompt 有两个来源

一道题实际收到哪套 system prompt，取决于该数据集 `tasks/<DS>/prompt.py` **是否提供 `build_system_prompt`**：

| 来源 | 触发条件 | 用它的数据集 |
|---|---|---|
| **A. `boxed_directive`**（`prompts.py`，主路径） | 数据集提供 `build_system_prompt` | **18/19**：Belief_R·BigToM·EmoBench·ExploreToM·FanToM·FictionalQA·HellaSwag·HiToM·PUB·SimpleToM·SocialBench·SocialIQA·TactfulToM·ToMBench·ToMChallenges·ToMQA·ToMato·ToMi |
| **B. `_SYSTEM_PROMPTS`**（`protocols.py`，协议默认兜底） | 数据集**没有** `build_system_prompt` → 回退 `system_prompt_for()` | **仅 SoMBench** |

> 换言之：**日常几乎所有数据集走的都是来源 A**，来源 B 现在只兜底 SoMBench 一家。两套措辞不同、且在 `open` 上**有意分叉**（见下）。

来源 A 的数据集里，system prompt = **`[可选的官方前言] + boxed_directive(lang, 题型, reasoning)`**：
- **纯 `boxed_directive`**（14 个）：system prompt 就是下表这一句。
- **官方前言 + `boxed_directive`**（4 个：BigToM / EmoBench / TactfulToM / ToMBench）：为复刻原论文，前面还有一段官方风格描述，末尾接下表这句。

### 来源 A：`boxed_directive`（题型 × 风格）

**英文（en）**

| 风格 | 题型 | boxed_directive |
|---|---|---|
| 裸答 | mcq_single | Give your answer as `\boxed{X}`, where X is the letter of the single best option. |
| 裸答 | mcq_multi | Give your answer as one `\boxed{}` containing every correct option letter, comma-separated, e.g. `\boxed{A,C}`. |
| 裸答 | mcq_grouped | For each question in order, give its answer as a separate `\boxed{X}`, e.g. `\boxed{A}` for question 1 then `\boxed{C}` for question 2. |
| 裸答 | open | Put your final answer as a short phrase inside `\boxed{}`, e.g. `\boxed{Paris}`. |
| 推理 | mcq_single | Reason step by step about the characters' mental states, then give your final answer as `\boxed{X}`, where X is the letter of the single best option. |
| 推理 | mcq_multi | Reason step by step, then give your final answer as one `\boxed{}` containing every correct option letter, comma-separated, e.g. `\boxed{A,C}`. |
| 推理 | mcq_grouped | Reason step by step, then for each question in order give its answer as a separate `\boxed{X}`, e.g. `\boxed{A}` for question 1 then `\boxed{C}` for question 2. |
| 推理 | open | Reason step by step, then put your final answer as a short phrase inside `\boxed{}`, e.g. `\boxed{Paris}`. |

**中文（zh）**

| 风格 | 题型 | boxed_directive |
|---|---|---|
| 裸答 | mcq_single | 请把答案放进 `\boxed{X}`,其中 X 是唯一最合适选项的字母。 |
| 裸答 | mcq_multi | 请把所有正确选项的字母放进同一个 `\boxed{}` 中,用英文逗号分隔,例如 `\boxed{A,C}`。 |
| 裸答 | mcq_grouped | 请按顺序对每一道问题各输出一个 `\boxed{X}`,例如第一问 `\boxed{A}`、第二问 `\boxed{C}`。 |
| 裸答 | open | 请把最终答案以简短短语放进 `\boxed{}` 中,例如 `\boxed{巴黎}`。 |
| 推理 | mcq_single | 请先一步步推理角色的心理状态,然后把最终答案放进 `\boxed{X}`,其中 X 是唯一最合适选项的字母。 |
| 推理 | mcq_multi | 请先一步步推理,然后把所有正确选项的字母放进同一个 `\boxed{}` 中,用英文逗号分隔,例如 `\boxed{A,C}`。 |
| 推理 | mcq_grouped | 请先一步步推理,然后按顺序对每一道问题各输出一个 `\boxed{X}`,例如第一问 `\boxed{A}`、第二问 `\boxed{C}`。 |
| 推理 | open | 请先一步步推理,然后把最终答案以简短短语放进 `\boxed{}` 中,例如 `\boxed{巴黎}`。 |

> 注意 `open` 在来源 A 是 **`\boxed{}` 短答**：这是为 `open_judge: f1` 的短答集（ExploreToM / FictionalQA / SocialBench / ToMChallenges）设的边界——否则推理协议下整段推理会把短答案的 token-F1 稀释到接近 0（f1 判分会先抽 `\boxed{}` 内容再算分）。**新增 open+f1 数据集务必提供 `build_system_prompt`**，只吃来源 B 会漏掉这个边界。

### 来源 B：`_SYSTEM_PROMPTS`（协议默认，仅 SoMBench 兜底）

自带 `You are a careful reader` 前言；与来源 A 的关键差别是 **`open` 是 free-text（不套 `\boxed{}`）**——因为唯一用它的 SoMBench Q4 走 `rubric` 长答案（≤1000 字自由作答，套 `\boxed{}` 反而违背 rubric 约束）。

**英文（en）**

| 风格 | 题型 | system prompt |
|---|---|---|
| 裸答 | mcq_single | You are a careful reader answering a multiple-choice theory-of-mind question. Read the story and the question carefully, then give your final answer in the format `\boxed{X}` where X is the letter of the single best option. |
| 裸答 | mcq_multi | You are a careful reader answering a multiple-choice theory-of-mind question. Read the story and the question carefully, then give your final answer as one `\boxed{}` containing every correct option letter, comma-separated, e.g. `\boxed{A,C}`. |
| 裸答 | open | You are a careful reader answering a question about a story. Read the story and the question carefully, then give your final answer text. |
| 推理 | mcq_single | You are a careful reader answering a multiple-choice theory-of-mind question. Think step by step about the mental states of the characters, then output your final answer in the format `\boxed{X}` where X is the letter of the single best option. Put your final `\boxed{X}` on the last line. |
| 推理 | mcq_multi | You are a careful reader answering a multiple-choice theory-of-mind question. Think step by step about the mental states of the characters, then output your final answer as one `\boxed{}` containing every correct option letter, comma-separated, e.g. `\boxed{A,C}`. Put your final `\boxed{}` on the last line. |
| 推理 | open | You are a careful reader answering a question about a story. Think step by step about the mental states of the characters, then give your final answer. Put your final answer on the last line. |

**中文（zh）**

| 风格 | 题型 | system prompt |
|---|---|---|
| 裸答 | mcq_single | 你是一个认真阅读的人,正在回答一道关于心理状态(theory-of-mind)的单选题。请仔细阅读故事和问题,然后给出最终答案,格式为 `\boxed{X}`,其中 X 是唯一最合适选项的字母。 |
| 裸答 | mcq_multi | 你是一个认真阅读的人,正在回答一道关于心理状态(theory-of-mind)的多选题。请仔细阅读故事和问题,然后给出最终答案:把所有正确选项的字母放进同一个 `\boxed{}` 中,用英文逗号分隔,例如 `\boxed{A,C}`。 |
| 裸答 | open | 你是一个认真阅读的人,正在回答一道关于故事的问题。请仔细阅读故事和问题,然后给出最终的答案文本。 |
| 推理 | mcq_single | 你是一个认真阅读的人,正在回答一道关于心理状态(theory-of-mind)的单选题。请一步步推理角色的心理状态,然后输出最终答案,格式为 `\boxed{X}`,其中 X 是唯一最合适选项的字母。把最终的 `\boxed{X}` 放在最后一行。 |
| 推理 | mcq_multi | 你是一个认真阅读的人,正在回答一道关于心理状态(theory-of-mind)的多选题。请一步步推理角色的心理状态,然后输出最终答案:把所有正确选项的字母放进同一个 `\boxed{}` 中,用英文逗号分隔,例如 `\boxed{A,C}`。把最终的 `\boxed{}` 放在最后一行。 |
| 推理 | open | 你是一个认真阅读的人,正在回答一道关于故事的问题。请一步步推理角色的心理状态,然后给出最终答案。把最终答案放在最后一行。 |

> `_SYSTEM_PROMPTS` 还含 `mcq_grouped` 两条（en/zh × 裸答/推理），措辞同来源 A 的 grouped，此处从略。

### 两点共性

> - `direct` 与 `direct_think` 共用「裸答」措辞，区别只在 `enable_thinking`（前者关、后者开）；`cot` 与 `del_tom` 共用「推理」措辞。
> - 无论走哪个来源，分工不变：**system prompt 只放答题风格 + 格式指令，user prompt 放故事/问题/选项内容**。不设 `protocol`（`protocol: null`）时 system prompt 为空，答题指令改由 user prompt 承载（`include_instruction=True`）。详见 [add_new_dataset.md](add_new_dataset.md)。

---

## extractor（答案提取）

| extractor | 用于 | 规则 |
|---|---|---|
| `extract_direct` | `direct` | 取**第一个** `\boxed{}` 内容；没有则取全文第一个字母 |
| `extract_cot` | `direct_think` / `cot` / `del_tom` | 取**最后一个** `\boxed{}` 内容；没有则取末 200 字符内最后一个字母 |
| `legacy` | 不设协议 | 取最后一个 `\boxed{}`（历史行为） |

设计动机：裸答应当一上来就给答案（取第一个），推理把结论放在末尾（取最后一个）。
选择题严格判分——提取不到 `\boxed{}` 且无兜底字母时记为 `extraction_failed`。

---

## del_tom 多数投票

`del_tom` 复用「重复评测」机制对同一道题跑 `n_samples=8` 次，并**关闭选项打乱**
（8 次选项顺序一致，按字母投票才有意义；其余协议正常打乱）。投票在判分（metric）阶段进行：

- **mcq_single（主路径）**：每次用 `extract_cot` 取字母 → 多数投票，**平局取字母序最小**；8 次全部提取失败记为 `extraction_failed`。
- **mcq_multi（退化）**：逐字母严格多数（某字母出现在 > 半数 repeat 的答案集合中才计入），结果集合与正确集合比对。
- **open（退化）**：文本投票无良好定义，取 `repeat` 最小的代表样本走一次正常 LLM 判分。

投票把同一 sample 的 8 条预测**折叠成 1 条**再判分，因此最终统计的题数仍是真实题数（不是 8 倍），
各数据集的 `tasks/<DS>/metrics.py` 完全不用感知协议或投票。

---

## shuffle（选项打乱）

不再是配置开关，由协议决定：

| 协议 | shuffle |
|---|---|
| direct / direct_think / cot | 开（确定性种子 `dataset\|sample_id\|repeat`，可复现地打乱，减小位置偏置） |
| del_tom | 关（保证 8 次投票字母对齐） |
| 不设协议 | 开（旧行为） |

---

## 用法示例

```yaml
# experiment_config.yaml
llm:
  model_name: Qwen3-8B
  api_key: not-needed
  api_url: http://0.0.0.0:8000/v1
  max_workers: 16

protocol: del_tom      # 改这一行切换协议
stage: all
datasets:
  - BigToM
  - SocialIQA
max_samples: 0
```

```bash
python run_eval.py                       # 批量跑 datasets 列表
python tasks/SocialIQA/run.py            # 单跑一个数据集（同样从配置读 protocol/stage）
```
