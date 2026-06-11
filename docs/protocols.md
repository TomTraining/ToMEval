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

协议把答题格式指令从 user prompt **移到了 system prompt**，按「风格 × 题型 × 语言」生成：

- 风格只有两类：
  - **裸答**（`direct` / `direct_think`）：直接给最终答案，不要任何解释/推理。
  - **推理**（`cot` / `del_tom`）：一步步推理角色心理状态，最终答案放最后一行。
- 题型三类：`mcq_single`（单选）/ `mcq_multi`（多选）/ `open`（开放）。
- 语言两套：英文 `en` / 中文 `zh`（跟随样本 `meta.lang` / `meta.language`）。

### 英文（en）

| 风格 | 题型 | system prompt |
|---|---|---|
| 裸答 | mcq_single | You are a careful reader answering a multiple-choice theory-of-mind question. Read the story and the question carefully, then output ONLY your final answer in the format `\boxed{X}` where X is the letter of the single best option. Do not include any explanation, reasoning, or extra text. |
| 裸答 | mcq_multi | You are a careful reader answering a multiple-choice theory-of-mind question. Read the story and the question carefully, then output ONLY your final answer as one `\boxed{}` containing every correct option letter, comma-separated, e.g. `\boxed{A,C}`. Do not include any explanation, reasoning, or extra text. |
| 裸答 | open | You are a careful reader answering a question about a story. Read the story and the question carefully, then output ONLY the final answer text. Do not include any explanation, reasoning, or extra text. |
| 推理 | mcq_single | You are a careful reader answering a multiple-choice theory-of-mind question. Think step by step about the mental states of the characters, then output your final answer in the format `\boxed{X}` where X is the letter of the single best option. Put your final `\boxed{X}` on the last line. |
| 推理 | mcq_multi | You are a careful reader answering a multiple-choice theory-of-mind question. Think step by step about the mental states of the characters, then output your final answer as one `\boxed{}` containing every correct option letter, comma-separated, e.g. `\boxed{A,C}`. Put your final `\boxed{}` on the last line. |
| 推理 | open | You are a careful reader answering a question about a story. Think step by step about the mental states of the characters, then give your final answer. Put your final answer on the last line. |

### 中文（zh）

| 风格 | 题型 | system prompt |
|---|---|---|
| 裸答 | mcq_single | 你是一个认真阅读的人,正在回答一道关于心理状态(theory-of-mind)的单选题。请仔细阅读故事和问题,只输出最终答案,格式为 `\boxed{X}`,其中 X 是唯一最合适选项的字母。不要包含任何解释、推理或多余文字。 |
| 裸答 | mcq_multi | 你是一个认真阅读的人,正在回答一道关于心理状态(theory-of-mind)的多选题。请仔细阅读故事和问题,只输出最终答案:把所有正确选项的字母放进同一个 `\boxed{}` 中,用英文逗号分隔,例如 `\boxed{A,C}`。不要包含任何解释、推理或多余文字。 |
| 裸答 | open | 你是一个认真阅读的人,正在回答一道关于故事的问题。请仔细阅读故事和问题,只输出最终的答案文本。不要包含任何解释、推理或多余文字。 |
| 推理 | mcq_single | 你是一个认真阅读的人,正在回答一道关于心理状态(theory-of-mind)的单选题。请一步步推理角色的心理状态,然后输出最终答案,格式为 `\boxed{X}`,其中 X 是唯一最合适选项的字母。把最终的 `\boxed{X}` 放在最后一行。 |
| 推理 | mcq_multi | 你是一个认真阅读的人,正在回答一道关于心理状态(theory-of-mind)的多选题。请一步步推理角色的心理状态,然后输出最终答案:把所有正确选项的字母放进同一个 `\boxed{}` 中,用英文逗号分隔,例如 `\boxed{A,C}`。把最终的 `\boxed{}` 放在最后一行。 |
| 推理 | open | 你是一个认真阅读的人,正在回答一道关于故事的问题。请一步步推理角色的心理状态,然后给出最终答案。把最终答案放在最后一行。 |

> 注：`direct` 与 `direct_think` 共用「裸答」prompt，区别只在 `enable_thinking`（前者关、后者开）。
> `cot` 与 `del_tom` 共用「推理」prompt。

> **题型补充**：除上面三类，还有 `mcq_grouped`——一个 prompt 内含多道子问题（如 EmoBench EU 的情绪+原因），
> 由数据集的 `prepare_samples` 钩子预先打包，按规则判分（每个子问题各对才算整体对）。
>
> **数据集级覆盖**：若数据集需忠实复刻原论文的 system prompt（如 ToMBench/EmoBench/FanToM），
> 可在 `tasks/<数据集>/prompt.py` 提供 `build_system_prompt(sample, protocol, lang, prompt_type)`，
> 覆盖上表的通用 system prompt（仍把答案格式统一换成 `\boxed{}`）。user prompt 同理可由 `build_prompt` 覆盖。
> 详见 [add_new_dataset.md](add_new_dataset.md)。无论是否覆盖，分工不变：**system prompt 只放答题风格 + 格式指令，user prompt 放故事/问题/选项内容**。

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
