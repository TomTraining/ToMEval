# 数据集总览（Datasets）

本文件介绍 ToMEval 当前内置的 **19 个数据集**：它们各自考察什么能力、从哪个原始数据集**改造**而来、以及框架**如何评测**它们。

- 想新增一个数据集 → 看 [add_new_dataset.md](add_new_dataset.md)
- 想了解四个评测协议（direct / direct_think / cot / del_tom）→ 看 [protocols.md](protocols.md)
- 想了解出图 → 看 [visualization.md](visualization.md)

---

## 1. 一句话定位

所有数据集都被归一化成同一套 schema（见 [§4](#4-如何改造原始数据--标准化-schema)），跑同一条评测流水线（见 [§5](#5-如何评测评测流水线)），因此被测模型在 19 个数据集上的表现可以横向对比。

按考察能力分三类：

| 类别 | 数据集 | 说明 |
|------|--------|------|
| **ToM / 社会认知**（主） | BigToM · EmoBench · FanToM · HiToM · ToMBench · TactfulToM · SimpleToM · Belief_R · ExploreToM · ToMato · ToMQA · ToMi · ToMChallenges | 错误信念、高阶信念、情绪、意图、得体性、信念修正等 |
| **社会智能 / 语用** | SocialIQA · PUB · SocialBench | 社会常识、语用理解（反讽 / 言外之意）、角色扮演社交 |
| **常识对照（非 ToM）** | HellaSwag · FictionalQA | 作为对照基线：句子续写常识 / 虚构文档阅读，用于区分"会读题"和"会读心" |
| **混合长题型（含 LLM judge）** | SocialToM | socmind v5.3，单条样本含 Q1–Q4 四种题型，Q4 开放分析题走 rubric LLM 判分 |

---

## 2. 速查表

| 数据集 | 原始来源 | 样本量 | 题型 | 判分 | 复刻原论文 prompt |
|--------|----------|-------:|------|------|:---:|
| BigToM | cicl-stanford/procedural-evals-tom (ICML'24) | 5,000 | mcq_single | 规则 | ✓ |
| EmoBench | Sahandfer/EmoBench (ACL'24) | 1,200 | mcq_grouped / single | 规则 | ✓ |
| FanToM | skywalker023/fantom (EMNLP'23) | 11,292 | mcq_single | 规则 | ✓ |
| HiToM | ying-hui-he/Hi-ToM (EMNLP'23) | 1,200 | mcq_single(15选) | 规则 | ✓ |
| SocialIQA | allenai/social_i_qa (EMNLP'19) | 1,954 | mcq_single(3选) | 规则 | ✓ |
| ToMBench | zhchen18/ToMBench (ACL'24) | 5,720 | mcq_single | 规则 | ✓ |
| TactfulToM | nii-cl/tactful-tom (EMNLP'25) | 4,254 | mcq_single / multi | 规则 | ✓ |
| SimpleToM | allenai/SimpleToM (arXiv 2410) | 3,441 | mcq_single | 规则 | ✓ |
| Belief_R | CAiRE/belief_r (EMNLP'24) | 3,656 | mcq_single | 规则 | ✓ |
| ExploreToM | facebook/ExploreToM (ICLR'25) | 13,309 | mcq_single + **open** | 规则 + **f1** | ✓ |
| PUB | cfilt/PUB (ACL'24 Findings) | 26,645 | mcq_single | 规则 | ✓ |
| HellaSwag | Rowan/hellaswag (ACL'19) | 10,042 | mcq_single(4选) | 规则 | ✓ |
| FictionalQA | tomg-group-umd/fictionalqa (arXiv 2506) | 7,499 | **open** | **f1** | ✓ |
| ToMato | nttmdlab-nlp/ToMATO (AAAI'25) | 5,401 | mcq_single(4选) | 规则 | ✓ |
| SocialBench | X-PLUG/SocialBench (ACL'24 Findings) | 7,701 | mcq_single + **open** | 规则 + **f1** | ✓ |
| ToMChallenges | xiaomeng-ma/ToMChallenges (CoNLL'23) | 720 | mcq_single + **open** | 规则 + **f1** | ✓ |
| ToMQA | kayburns/tom-qa-dataset (EMNLP'18) | 12,000 | mcq_single | 规则 | ✓ |
| ToMi | facebookresearch/ToMi (EMNLP'19) | 5,638 | mcq_single | 规则 | ✓ |
| SocialToM | socmind v5（内部） | 1,704 | single / multi / **open** | 规则 + **rubric** | （通用模板） |

> 样本量为 `datasets/<DS>/*.parquet` 实际行数。"复刻原论文 prompt" 指 `tasks/<DS>/prompt.py` 提供了忠实题面排版的钩子（见 [§5.2](#52-prompt-构造)）；SocialToM 走框架通用模板。

---

## 3. 各数据集详情

每条按 **来源 → 考察能力 → 题型 → 改造（`scripts/convert_*.py`）→ 判分** 组织。

### 3.1 ToM / 社会认知

#### BigToM — 因果模板生成的一阶信念
- **来源**：cicl-stanford/procedural-evals-tom，《Understanding Social Reasoning in LLMs with LLMs》(Gandhi et al., ICML 2024)，GPT-4 按因果模板程序化生成。
- **能力**：前向信念（由感知推断信念）、前向行动（由感知预测行为）、后向信念（由行为反推初始信念），含 true/false belief 与 desire 维度。
- **题型**：`mcq_single`。
- **改造**：原始已是标准结构，直接归一化为 story/question/correct/wrong/meta（无独立 convert 脚本）。
- **判分**：规则（`\boxed{}` 抽取 + 字母匹配）。

#### EmoBench — 情绪智能（英中双语）
- **来源**：Sahandfer/EmoBench，《EmoBench》(Sabour et al., ACL 2024)，400 个手工场景。
- **能力**：情绪理解（EU：识别情绪 + 分析成因，两个子问）、情绪应用（EA：推荐有效应对）。
- **题型**：EU → `mcq_grouped`（两问都对才算对），EA → `mcq_single`。
- **改造**：原始格式直接标准化；`tasks/EmoBench/prompt.py` 的 `prepare_samples` 钩子在预测前把 EU 的两个子问合并为一条 grouped 多问。
- **判分**：规则；grouped 要求每个子问各对。

#### FanToM — 对话信念 + 信息可达性
- **来源**：skywalker023/fantom（经 TomTraining/PureToMDatasets 中转），来自虚构小说对话 + ToM 标注。
- **能力**：事实 QA、信念 QA、可答性（问题是否可答）、信息可达性（角色能否获得某信息）。
- **题型**：`mcq_single`（含 yes/no 二元）。
- **改造**（`convert_fantom.py`）：按 `set_id` 从原始 `fantom_v1.json` 取回 `fact_question / fact_answer / short_context` 三字段补进 meta（行数与答案编码不变）；按题型拼前缀（Answerability 用 `Target:`，Info-Accessibility 用 `Information:`），选项编码 A/B/C/D。
  > ⚠️ 这三个字段由 convert 脚本补回，会被 `download_datasets.sh` 覆盖，重新下载后需重跑该脚本。
- **判分**：规则。

#### HiToM — 0–4 阶高阶信念
- **来源**：ying-hui-he/Hi-ToM_dataset，《Hi-ToM》(Wu et al., EMNLP 2023)，Sally-Anne 变体的多智能体寻物游戏。
- **能力**：0 阶（物体真实位置）到 4 阶（嵌套信念"A 认为 B 认为…"），含 Tell/No-Tell 欺骗变体。
- **题型**：`mcq_single`（15 选一）。
- **改造**：保留原始 prompt 字段；`tasks/HiToM/prompt.py`（`prompt_style: hitom`）在题面后固定追加官方 4 条假设 Note。
- **判分**：规则。

#### ToMBench — 31 项社会认知能力（双语，防泄露）
- **来源**：zhchen18/ToMBench，《ToMBench》(Chen et al., ACL 2024)，从零构建避免训练集泄露，ATOMS 框架 8 任务 × 31 能力 × 6 维度。
- **能力**：情绪、意图、信念、行为预测、情感调节等。
- **题型**：`mcq_single`。
- **改造**：原始格式直接标准化；`tasks/ToMBench/prompt.py` 按中英双语分别给 system prompt。
- **判分**：规则。

#### TactfulToM — 得体性 / 善意谎言
- **来源**：nii-cl/tactful-tom，《TactfulToM》(EMNLP 2025)。**License 仅供评测、禁止训练。**
- **能力**：在对话中理解善意谎言、敏感信息、人物关系、情感语境（谎言可检测性 / 可接受性）。
- **题型**：二元题与单答题 → `mcq_single`，"列出所有角色"题 → `mcq_multi`。
- **改造**（`convert_tactfultom.py`）：每条对话作 story 展开为多行（每类问题一行）；二元题合成相反项作干扰，list 题聚合为多选。
- **判分**：规则。

#### SimpleToM — 显式推理 vs 隐式应用
- **来源**：allenai/SimpleToM，《SimpleToM》(Gu et al., arXiv 2410.13648)，三个 QA 子集。
- **能力**：心理状态推理（信息可达）、行为预测、社会判断——揭示"会推断心理状态"与"会据此判断行为"之间的落差。
- **题型**：`mcq_single`。
- **改造**（`convert_simpletom.py`）：三个子集统一映射为单选，`answerKey` 指向正确项，其余为干扰；用 `meta.qa_type / meta.dimension` 区分子集。
- **判分**：规则。

#### Belief_R — 信念修正
- **来源**：CAiRE/belief_r，《Belief Revision》(EMNLP 2024)，Delta-Reasoning 方法。
- **能力**：新前提下判断必然结论，分 time_t（初始推理）与 time_t1（信念更新/维持）两阶段。
- **题型**：`mcq_single`（(a)/(b)/(c)）。
- **改造**（`convert_belief_r.py`）：把 questions 拆成 story（前提）+ question（推理问题），`ground_truth` 指向正确项，`meta.step` 区分两阶段。
- **判分**：规则。

#### ExploreToM — 对抗生成的高阶信念
- **来源**：facebook/ExploreToM，《Explore Theory of Mind》(Sclar et al., ICLR 2025)，程序+对抗生成。
- **能力**：角色对物体位置/知晓状态的信念，含 1 阶、1–2 阶混合错误信念。
- **题型**：yes/no 与 knows/doesn't-know → `mcq_single`；**位置题（无干扰项）→ `open`**。
- **改造**（`convert_exploretom.py`）：按 `expected_answer` 形态判题型（yes/no、knows、其余为容器名走 open），`meta` 记录 `nth_order / is_false_belief_1st` 等。
- **判分**：二元题规则；位置题 **token-F1**（`open_judge: f1`）。

#### ToMato — 角色扮演下的心理状态
- **来源**：nttmdlab-nlp/ToMATO，《ToMATO》(AAAI 2025)。**License 仅评测、禁止训练。**
- **能力**：由角色扮演 LLM 生成的对话，推断说话者（高阶）心理状态（情绪、意图等）。
- **题型**：`mcq_single`（4 选一）。
- **改造**（`convert_tomato.py`）：conversation → story，4 个候选 a0..a3，`a_idx` 指向正确项。
- **判分**：规则。

#### ToMQA — bAbI 格式的信念 QA
- **来源**：kayburns/tom-qa-dataset，《Evaluating Theory of Mind in Question Answering》(Nematzadeh et al., EMNLP 2018)。
- **能力**：true belief / false belief / 二阶 false belief；信息类型分 memory / reality（控制）/ belief / search（ToM）。
- **题型**：`mcq_single`（从故事中出现的容器里选）。
- **改造**（`convert_tomqa.py`）：解压 bAbI 文本，按行号切故事块，陈述句作前文、含 `?` 行作问题，答案为容器名、其余容器作干扰；文件名解析出 qtype。
- **判分**：规则。

#### ToMi — 一阶/二阶错误信念（生成式）
- **来源**：facebookresearch/ToMi，《Revisiting the Evaluation of ToM through QA》(Le et al., EMNLP 2019)，脚本生成。
- **能力**：一阶/二阶 false belief、true belief、物体位置与角色信念。
- **题型**：`mcq_single`（容器选择）。
- **改造**（`convert_tomi.py`）：读 `test.txt` + `test.trace`，按行号切块，从 trace 提取 `question_type / story_type`（true/false/second_order）。
- **判分**：规则。

#### ToMChallenges — 经典 Sally-Anne / Smarties
- **来源**：xiaomeng-ma/ToMChallenges，《ToMChallenges》(CoNLL 2023)，原理导向、多答题格式。
- **能力**：错误信念（Sally-Anne）、物体替换（Smarties），多种题面格式。
- **题型**：mc 格式 → `mcq_single`（A/B）；qa 格式 → `open`。
- **改造**（`convert_tomchallenges.py`）：读 CSV，用正则从 `mc_prompt` 抽 A/B 选项，qa 取 `short_answer` 作开放答案，`meta.task_format` 区分。
- **判分**：mc 规则；qa **token-F1**。

### 3.2 社会智能 / 语用

#### SocialIQA — 社会常识
- **来源**：allenai/social_i_qa，《Social IQa》(Sap et al., EMNLP 2019)，基于 ATOMIC 知识图谱。
- **能力**：理解行为的社会含义（非物理常识），9 维社会推理。
- **题型**：`mcq_single`（3 选一）。
- **改造**：直接标准化（社区约定的 Context/Question/Answers 排版）。
- **判分**：规则。

#### PUB — 语用理解
- **来源**：cfilt/PUB，《PUB: Pragmatics Understanding Benchmark》(ACL 2024 Findings)，14 个语用子任务合并。
- **能力**：反讽检测、言外之意（implicature）等语用能力（`meta.task_type` 标 14 类）。
- **题型**：`mcq_single`。
- **改造**（`convert_pub.py`）：`pretext → story`，`options` 作候选；归一化文本后把 correct 映射回原始选项；过滤无 options 或 correct 不在 options 中的样本。
- **判分**：规则。

#### SocialBench — 角色扮演社交（双语）
- **来源**：X-PLUG/SocialBench，《SocialBench》(ACL 2024 Findings)，4 子集、中英双语。
- **能力**：情感感知、自我认知、社交偏好、对话记忆。
- **题型**：有 choices → `mcq_single`；conversation_memory（无 choices）→ `open`。
- **改造**（`convert_socialbench.py`）：`profile + dialogue → story`，`instruction → question`；有 choices 走单选，无 choices 把关键词拼成 correct 走 open，`meta.lang` 标语言。
- **判分**：单选规则；记忆题 **token-F1**。

### 3.3 常识对照（非 ToM）

#### HellaSwag — 句子续写常识
- **来源**：Rowan/hellaswag，《HellaSwag》(Zellers et al., ACL 2019)。**作对照基线，非 ToM。**
- **能力**：给定上文选最合理续写（物理/常识，不涉及心理状态）。
- **题型**：`mcq_single`（4 选一）。
- **改造**（`convert_hellaswag.py`）：用 validation split（test 标签未公开），`ctx → story`，4 个 endings 作选项，`label` 指向正确续写，`activity_label` 进 meta。
- **判分**：规则。

#### FictionalQA — 虚构文档阅读
- **来源**：tomg-group-umd/fictionalqa（joined_qa 子集），arXiv 2506.05639。**作对照基线，非 ToM。**
- **能力**：开卷阅读理解 / 知识获取（给完整虚构文档问事实细节）。
- **题型**：`open`（无干扰项）。
- **改造**（`convert_fictionalqa.py`）：`fiction → story`，`natural_answer → correct_answers`，wrong 为空；过滤空答案 / `unknown_answer`。
- **判分**：**token-F1**（`open_judge: f1`）。

### 3.4 混合长题型

#### SocialToM — socmind v5（Q1–Q4 + rubric judge）
- **来源**：项目内 `dataset_v5/`（dataset_v5_1/2/3.json 三批合并，共 284 样本），socmind-bench / SocialToM 体系。是旧 V4p2 的新版数据。
- **规模**：284 样本 × 6 题/样本 = **1,704 行**；覆盖一级维度 1/2/3、共 71 个三级维度（task_id）。
- **能力**：社会规范 / 文化理解，一条样本含四种题：
  - Q1 单选（`mcq_single`）、Q2 不定项（`mcq_multi`）、Q3 多个"是/否/无法确定"判断子项（每子项 `mcq_single`）、**Q4 开放分析长答（`open`）**。
- **改造**（`convert_socialtom.py`）：合并 `dataset_v5/` 三批、按 id 去重；把每个样本的 Q1–Q4 展开为独立行，各自带 story/question/answer；兼容 `题目` 的 list/dict 两种形态、Q2 答案兼容 `answer_key.Q2` 与 `题目.Q2.答案`；Q4 的 correct 为参考要点串、wrong 为空；`meta` 固定 keyset（`id / dim / dim1 / dim2 / qtype / perspective / variant / length_mode / lang / q4_reference` 等）保证 parquet schema 一致。
- **判分**：Q1–Q3 规则；**Q4 走 rubric LLM judge**（`open_judge: rubric`）：按 `meta.dim` 选 `q4_judge_prompts.json` 里的专属评分 prompt，judge 给 0–10 总分（D1–D5 各 0–2），平均分 ≥ `open_threshold`(7.0) 记为正确；可选 `judge2` 双 judge 取平均。judge 模型在 `tasks/SocialToM/config.yaml` 的 `judge1/judge2` 段配置，与被测模型解耦。

---

## 4. 如何改造（原始数据 → 标准化 schema）

每个数据集对应一个 `scripts/convert_<name>.py`（少数原始即标准格式的无需脚本），统一输出 `datasets/<DS>/test-*.parquet`，schema 固定为：

```json
{
  "story":    "背景故事 / 对话 / 文档",
  "question": "问题文本",
  "answer": {
    "correct_answers": ["正确答案", "..."],   // 永远是字符串列表
    "wrong_answers":   ["干扰项1", "干扰项2"]  // 为空 → 开放题
  },
  "meta": { "id": "...", "dimension": "...", "lang": "...", "...": "数据集专属分组字段" }
}
```

**改造时的通用动作**：
1. **抽取三要素**：把原始字段拆成 `story` / `question` / 答案。
2. **构造选项**：单选取正确项 + 干扰项；二元题（yes/no、knows/doesn't）合成相反项作干扰；列举题聚合为多选；无干扰项的短答 / 长答留空 `wrong_answers` → 自动判为开放题。
3. **映射答案**：把原始 answer key / index / 文本归一化后对应到 `correct_answers`。
4. **过滤**：丢弃缺选项、答案不在选项内、空答案等脏样本。
5. **写 meta**：保留分组维度（能力维度、子集、语言、题型、阶数等），供分组指标与可视化使用。

**题型自动判定**（无需手工标注，见 `src/evaluation/prompts.py`）：

| 条件 | 题型 |
|------|------|
| `wrong_answers` 为空 且 唯一正确答案 | `open`（开放题，judge 判分） |
| 多个正确答案 | `mcq_multi`（多选，`\boxed{A,C}`） |
| 否则 | `mcq_single`（单选，`\boxed{X}`） |
| `prepare_samples` 钩子合并多个子问 | `mcq_grouped`（多问，每问各一 `\boxed{}`，全对才算对） |

下载/重建数据：`download_datasets.sh`（注意 FanToM 需重跑 `convert_fantom.py` 补字段）。

---

## 5. 如何评测（评测流水线）

入口 `python run_eval.py`，逐个数据集跑 `tasks/<DS>/run.py`；阶段由 `experiment_config.yaml` 的 `stage` 控制（`predict` / `metric` / `visualize` / `all`）。完整流程：

```
加载标准化数据 → (可选)prepare_samples 预处理 → 按协议/prompt.py 构造 Prompt
   → 并发调用被测模型 → 抽取答案 → 判分 → (del_tom)多数投票 → 聚合 metrics → 出图
```

### 5.1 评测协议（`src/evaluation/protocols.py`）

采样参数、system prompt 风格、答案抽取、投票全部由 `protocol` 驱动：

| 协议 | temperature | enable_thinking | n_samples | system prompt | 答案抽取 |
|------|:-----------:|:---------------:|:---------:|---------------|----------|
| `direct` | 0.0 | False | 1 | 裸答（直接给 `\boxed{}`） | 第一个 `\boxed{}` |
| `direct_think` | 0.6 | True | 1 | 裸答 | 第一个 `\boxed{}` |
| `cot` | 0.6 | True | 1 | 逐步推理，末行 `\boxed{}` | 最后一个 `\boxed{}` |
| `del_tom` | 0.6 | True | 8 | 逐步推理 | 最后一个 `\boxed{}` + 多数投票 |

（top_p=0.95、max_tokens=32768 四协议一致；`del_tom` 跑 8 次且不 shuffle 选项，按字母多数投票。详见 [protocols.md](protocols.md)。）

### 5.2 Prompt 构造

- **默认**：框架按 `协议风格 × 语言 × 题型` 生成统一的 system prompt + 题面。
- **复刻原论文**（可选）：`tasks/<DS>/prompt.py` 提供 `build_prompt` / `build_system_prompt` / `prepare_samples` 钩子即可忠实还原原论文排版（约定式加载，缺省回退通用实现）。目前除 SocialToM 外的 18 个数据集都接入了 `prompt.py`。

### 5.3 判分

- **选择题 / grouped 多问**：规则判分——抽 `\boxed{}` 里的字母与正确答案比对，**不需要 judge 模型**。
- **开放题**：判分方式由数据集自己在 `tasks/<DS>/config.yaml` 的 `open_judge` 字段决定（见 `src/evaluation/open_judge.py`）：

| `open_judge` | 机制 | 是否需 judge 模型 | 用在 |
|--------------|------|:---:|------|
| `f1` | 预测与参考的 token/char 级 F1，按 `f1_threshold`(默认 0.5) 二值化 | 否 | ExploreToM·FictionalQA·SocialBench·ToMChallenges 的开放子集 |
| `llm_simple` | 二元 LLM judge（"答案对不对"） | 是（`judge1`） | 默认开放题模式 |
| `rubric` | 按维度专属 rubric 给 0–max_score 总分，均分 ≥ `open_threshold` 算对 | 是（`judge1`，可选 `judge2` 取平均） | SocialToM Q4 |

judge 模型在数据集自己的 `config.yaml` 里配置（`judge1/judge2`），与 `experiment_config.yaml` 里的被测模型完全解耦。

### 5.4 指标聚合（`src/evaluation/task_metrics.py`）

每个数据集的 `tasks/<DS>/metrics.py` 产出：

- **基础指标**：`accuracy`、`correct`、`total`、`extraction_failed`（`\boxed{}` 抽取失败数）/`extraction_failed_rate`。
- **分组指标**：`by_<x>`（如 `by_dimension` / `by_qtype` / `by_lang`）+ 对应 `<x>_counts`，由 `generic_group_metrics` 按 `meta` 字段声明。组合键（值形如 `"a|b"`，如 SocialToM 的 `by_dim3_qtype`）可视化时自动透视成热力图。
- **分组均分**：非 0-1 的均分字典（如 SocialToM 的 `q4_mean_score_by_dim`，0–10 分）。
- **组级全对指标**：要求一组样本全部答对才算通过（如多成员子问）。
- 多轮（`repeats`）取平均存入 `avg_metrics`，单轮明细存 `all_metrics`。

结果落在 `results/<DS>/<模型>/exp_<时间戳>/`（`prediction.jsonl` + `metrics.json`），出图见 [visualization.md](visualization.md)，汇总成表见 README「生成对比表格」。

---

## 6. 当前启用清单

`experiment_config.yaml` 的 `datasets` 列表控制批量评测哪些数据集。19 个数据集全部就绪（`datasets/<DS>/` 有 parquet、`tasks/<DS>/` 有 `config.yaml`+`run.py`），可按需启用。

> 历史数据集 **V4p2**（socmind v4）已被其新版 **SocialToM**（socmind v5.3）取代并从仓库移除。
