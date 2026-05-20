# 数据合成迭代日志

每次运行 `run_data_processing.py` 后自动追加新节。
人工在每节末尾填写 **Gaps** 和 **Next** 部分。

---

## Iteration 1 — 2026-05-20（首轮预运行）

**Stage**: all  
**Datasets**: ToMBench + BigToM  
**Synthesis model**: deepseek-v4-flash  
**samples_per_batch**: 5 | **samples_per_report**: 3 | **max_retries**: 3

### Stats

| Dataset  | bad_cases | reports | raw | clean | dropped |
|----------|-----------|---------|-----|-------|---------|
| ToMBench | 80        | 17      | 51  | 51    | 0       |
| BigToM   | 80        | 16      | 48  | 48    | 0       |

守门员测试集索引：25,230 条（7 个数据集），0 条泄漏。

### Samples

**ToMBench（修复后 iter2）**

- 故事：林悦和赵凯同事竞争总监职位，林悦偷取对方提案草稿。  
  问：林悦露出微笑最可能的心理？  
  正确：**她很高兴有机会看到赵凯的创意，做出针对性调整。**  
  干扰项：（A）好心收起避免丢失 （C）习惯性微笑 （D）觉得场面好笑 ✅ 高质量，意图推断

- 故事：林峰借阅赵雪参考材料，之后提案核心创意与赵雪草图惊人相似。  
  问：林峰借材料时真正的意图是什么？  
  正确：**想确认赵雪方向以便抢先利用。**  
  干扰项：真心借鉴 / 随口客套 / 测试信息共享意愿 ✅ 多层迷惑，难度高

**BigToM（iter2）**

- 故事：Ravi 在花园把水壶放好，去小屋休息，风把水壶吹翻。  
  问：Ravi 回来时认为水壶里有什么？  
  正确：Fresh water（他不知道被吹翻）  
  干扰项：mixture of water and plant food ✅ 标准 false_belief 形式，但场景简单

### Gaps（第一轮发现）

1. **ToMBench：正则解析失败（已修复）**  
   原始 iter1 中 `correct_answers` 只有字母 `['B']` 而非全文，原因是中文场景中 `\b` 不匹配非英文边界。已改用 `(?<![A-Za-z])` 负向后行断言。

2. **ToMBench：meta_id 重复（已修复）**  
   LLM 自行生成 `synthetic_0001`，多条相同。现在强制用 `r_idx/s_idx` 覆盖。

3. **ToMBench：语言不统一**  
   同一 batch 里有中文有英文样本，多样性可以接受，但若训练集需要纯中文版，可在 prompt 里加语言约束。

4. **BigToM：dimension 只有 `false_belief`，condition_type 多样性为零**  
   原因：bad cases 全来自 `backward_belief` condition（最难维度），但 Stage2 只用 `dimension` 字段分组，把所有 backward_belief bad case 归入同一 "false_belief" 报告，stage3 生成时也没有明确传入 `condition_type`，模型默认生成最简单的 `false_belief` 故事。  
   **BigToM 训练集实际有 6 种 condition_type**：forward_action(551), backward_belief(391), true_belief(270), percept_to_belief(251), forward_belief(239), false_belief(84)。合成数据必须覆盖全部类型。

5. **BigToM：故事场景重复（gardener watering can 多次出现）**  
   同一维度的多个 report 用了相似的模板故事，多样性不足。需要在 prompt 里明确列出已用场景，禁止重复。

6. **BigToM：condition_type 分组策略错误**  
   当前 `get_dimension_key` 对 BigToM 读 `dimension[0]`（`first_order`），而实际上 bad cases 的关键维度是 `condition_type`（`backward_belief`）。应以 `condition_type` 为分组键。

### Next（第二轮改进方向）

1. **修 BigToM 分组逻辑**：`get_dimension_key` 对 BigToM 优先返回 `condition_type`，使 Stage2 能按 6 种类型分别诊断和合成。
2. **扩展 BigToM bad case 覆盖**：当前只取了 `backward_belief` 的 bad case，需要对所有 6 种 condition_type 分别加载 bad case（可按 condition_type 分别限额）。
3. **合成 prompt 增加多样性约束**：在 `build_stage2_generation_from_report_prompt` 里加入"已用场景列表"，要求新故事与已有样本场景不重复。
4. **BigToM 合成 schema 补充 condition_type 字段**：让模型生成时明确指定 `condition_type`，而不是依赖 meta 手动写入。

---

## Iteration 2 — 2026-05-20（BigToM 分组修复 + 多样性增强，待执行）

见下一轮运行后追加。

---

## Iteration 3 — 2026-05-20 14:02
**Stage**: synth
**Datasets**: ToMBench
**Synthesis model**: deepseek-v4-flash
**samples_per_batch**: 5 | **samples_per_report**: 3 | **max_retries**: 3

### Stats
- **ToMBench**: bad_cases=? | reports=? | raw=51 | clean=51 | dropped=0
### Samples (auto)
- **ToMBench** — {"story": "在年终项目评审会上，林涛展示了他带领团队熬夜完成的季度报告。他的同事周敏注意到，当老板称赞报告的数据分析部分时，林涛短暂地瞥了一眼坐在角落的实习生小陈，嘴角微微上扬了一下。周敏知道，那份数据分析实际上是上周小陈私下给她看过的个人练习作品，林涛从未提过小陈的贡献。", "question": "周敏看到林涛瞥向小陈并微笑时，她最可能认为林涛的这个非言语动作意图是什么？A. 林涛在感谢小陈提供了数据，想用微笑表达认可。B. 林涛在确认小陈没有当众揭穿他窃取功劳的行为。C. 林涛在向小陈示意让他也参与讨论，以便分享荣誉。D. 林涛只是习惯性地扫视房间，微笑是因为项目成功了。",
- **ToMBench** — {"story": "在年度产品评审会上，陈经理展示了他主导的新款智能手表原型。同事李总监注意到手表的健康监测算法似乎直接复制了他团队去年未获批准的一个内部方案中的核心代码段。陈经理微笑着对李总监说：“你的团队之前做的市场调研真是帮了大忙，给了我很多启发。” 李总监回以微笑，但会后他悄悄把一份包含原始代码日期的文档副本发给了部门主管。", "question": "李总监为什么在会后把原始代码日期的文档副本发给部门主管？ A. 他真心感谢陈经理在评审会上的夸奖，想分享更多细节帮助陈经理改进产品。 B. 他想向部门主管证明陈经理的新款智能手表原型使用了李总监团队未获批准的方案，从而揭露抄袭行为。
- **ToMBench** — {"story": "In a competitive marketing department, Lina was vying with her colleague, Raj, for the coveted team lead position. Their manager, Mr. Chen, had asked everyone to submit peer feedback forms anonymously. Lina noticed Raj giving her a warm smile and a thumbs-up before the deadline, but she a

### Gaps
*(to be filled after manual review)*

### Next
*(to be filled after evaluation)*

---

## Iteration 3 — 2026-05-20 14:13
**Stage**: all
**Datasets**: BigToM
**Synthesis model**: deepseek-v4-flash
**samples_per_batch**: 5 | **samples_per_report**: 3 | **max_retries**: 3

### Stats
- **BigToM**: bad_cases=80 | reports=16 | raw=48 | clean=48 | dropped=0
### Samples (auto)
- **BigToM** — {"story": "Ravi is a gardener at a community garden. He carefully fills a large blue watering can with water and places it next to the tomato plants, planning to water them in the afternoon. While Ravi is taking a lunch break inside the shed, a sudden gust of wind knocks the watering can over, spill
- **BigToM** — {"story": "Leah is a florist. She carefully arranges a dozen red roses in a blue ceramic vase on the front counter, ready for a customer who ordered them. While Leah steps into the back room to get more ribbon, her colleague Sam decides to help by replacing the wilted roses with fresh ones, but he p
- **BigToM** — {"story": "Priya is a gardener at a community garden. She carefully fills a large watering can with water and places it next to the tomato plants, planning to water them after her lunch break. While Priya is inside eating, a sudden gust of wind knocks the watering can over, spilling all the water on

### Gaps
*(to be filled after manual review)*

### Next
*(to be filled after evaluation)*

---

## Iteration 1 — 2026-05-20 20:57
**Stage**: load
**Datasets**: BigToM
**Synthesis model**: deepseek-v4-flash
**samples_per_batch**: 5 | **samples_per_report**: 3 | **max_retries**: 3

### Stats
- **BigToM**: bad_cases=20 | reports=? | raw=? | clean=? | dropped=? | hard=?
### Samples (auto)
- **BigToM** — {"story": "Ravi is a gardener at a community garden. He carefully fills a large blue watering can with water and places it next to the tomato plants, planning to water them in the afternoon. While Ravi is taking a lunch break inside the shed, a sudden gust of wind knocks the watering can over, spill
- **BigToM** — {"story": "Leah is a florist. She carefully arranges a dozen red roses in a blue ceramic vase on the front counter, ready for a customer who ordered them. While Leah steps into the back room to get more ribbon, her colleague Sam decides to help by replacing the wilted roses with fresh ones, but he p
- **BigToM** — {"story": "Priya is a gardener at a community garden. She carefully fills a large watering can with water and places it next to the tomato plants, planning to water them after her lunch break. While Priya is inside eating, a sudden gust of wind knocks the watering can over, spilling all the water on

### Gaps
*(to be filled after manual review)*

### Next
*(to be filled after evaluation)*

---

## Iteration 1 — 2026-05-20 20:58
**Stage**: diagnose
**Datasets**: BigToM
**Synthesis model**: deepseek-v4-flash
**samples_per_batch**: 5 | **samples_per_report**: 3 | **max_retries**: 3

### Stats
- **BigToM**: bad_cases=3928 | reports=788 | raw=? | clean=? | dropped=? | hard=?
### Samples (auto)
- **BigToM** — {"story": "Ravi is a gardener at a community garden. He carefully fills a large blue watering can with water and places it next to the tomato plants, planning to water them in the afternoon. While Ravi is taking a lunch break inside the shed, a sudden gust of wind knocks the watering can over, spill
- **BigToM** — {"story": "Leah is a florist. She carefully arranges a dozen red roses in a blue ceramic vase on the front counter, ready for a customer who ordered them. While Leah steps into the back room to get more ribbon, her colleague Sam decides to help by replacing the wilted roses with fresh ones, but he p
- **BigToM** — {"story": "Priya is a gardener at a community garden. She carefully fills a large watering can with water and places it next to the tomato plants, planning to water them after her lunch break. While Priya is inside eating, a sudden gust of wind knocks the watering can over, spilling all the water on

### Gaps
*(to be filled after manual review)*

### Next
*(to be filled after evaluation)*

---

## Iteration 1 — 2026-05-20 21:14
**Stage**: load
**Datasets**: BigToM
**Synthesis model**: deepseek-v4-flash
**samples_per_batch**: 5 | **samples_per_report**: 3 | **max_retries**: 3

### Stats
- **BigToM**: bad_cases=20 | reports=? | raw=? | clean=? | dropped=? | hard=?
### Samples (auto)
- **BigToM** — {"story": "Ravi is a gardener at a community garden. He carefully fills a large blue watering can with water and places it next to the tomato plants, planning to water them in the afternoon. While Ravi is taking a lunch break inside the shed, a sudden gust of wind knocks the watering can over, spill
- **BigToM** — {"story": "Leah is a florist. She carefully arranges a dozen red roses in a blue ceramic vase on the front counter, ready for a customer who ordered them. While Leah steps into the back room to get more ribbon, her colleague Sam decides to help by replacing the wilted roses with fresh ones, but he p
- **BigToM** — {"story": "Priya is a gardener at a community garden. She carefully fills a large watering can with water and places it next to the tomato plants, planning to water them after her lunch break. While Priya is inside eating, a sudden gust of wind knocks the watering can over, spilling all the water on

### Gaps
*(to be filled after manual review)*

### Next
*(to be filled after evaluation)*

---

## Iteration 1 — 2026-05-20 21:15
**Stage**: diagnose
**Datasets**: BigToM
**Synthesis model**: deepseek-v4-flash
**samples_per_batch**: 5 | **samples_per_report**: 3 | **max_retries**: 3

### Stats
- **BigToM**: bad_cases=? | reports=7 | raw=? | clean=? | dropped=? | hard=?
### Samples (auto)
- **BigToM** — {"story": "Ravi is a gardener at a community garden. He carefully fills a large blue watering can with water and places it next to the tomato plants, planning to water them in the afternoon. While Ravi is taking a lunch break inside the shed, a sudden gust of wind knocks the watering can over, spill
- **BigToM** — {"story": "Leah is a florist. She carefully arranges a dozen red roses in a blue ceramic vase on the front counter, ready for a customer who ordered them. While Leah steps into the back room to get more ribbon, her colleague Sam decides to help by replacing the wilted roses with fresh ones, but he p
- **BigToM** — {"story": "Priya is a gardener at a community garden. She carefully fills a large watering can with water and places it next to the tomato plants, planning to water them after her lunch break. While Priya is inside eating, a sudden gust of wind knocks the watering can over, spilling all the water on

### Gaps
*(to be filled after manual review)*

### Next
*(to be filled after evaluation)*

---

## Iteration 1 — 2026-05-20 21:16
**Stage**: diagnose
**Datasets**: BigToM
**Synthesis model**: deepseek-v4-flash
**samples_per_batch**: 5 | **samples_per_report**: 3 | **max_retries**: 3

### Stats
- **BigToM**: bad_cases=? | reports=7 | raw=? | clean=? | dropped=? | hard=?
### Samples (auto)
- **BigToM** — {"story": "Ravi is a gardener at a community garden. He carefully fills a large blue watering can with water and places it next to the tomato plants, planning to water them in the afternoon. While Ravi is taking a lunch break inside the shed, a sudden gust of wind knocks the watering can over, spill
- **BigToM** — {"story": "Leah is a florist. She carefully arranges a dozen red roses in a blue ceramic vase on the front counter, ready for a customer who ordered them. While Leah steps into the back room to get more ribbon, her colleague Sam decides to help by replacing the wilted roses with fresh ones, but he p
- **BigToM** — {"story": "Priya is a gardener at a community garden. She carefully fills a large watering can with water and places it next to the tomato plants, planning to water them after her lunch break. While Priya is inside eating, a sudden gust of wind knocks the watering can over, spilling all the water on

### Gaps
*(to be filled after manual review)*

### Next
*(to be filled after evaluation)*

---

## Iteration 1 — 2026-05-20 21:25
**Stage**: diagnose
**Datasets**: BigToM
**Synthesis model**: deepseek-v4-flash
**samples_per_batch**: 5 | **samples_per_report**: 3 | **max_retries**: 3

### Stats
- **BigToM**: bad_cases=? | reports=7 | raw=? | clean=? | dropped=? | hard=?
### Samples (auto)
- **BigToM** — {"story": "Ravi is a gardener at a community garden. He carefully fills a large blue watering can with water and places it next to the tomato plants, planning to water them in the afternoon. While Ravi is taking a lunch break inside the shed, a sudden gust of wind knocks the watering can over, spill
- **BigToM** — {"story": "Leah is a florist. She carefully arranges a dozen red roses in a blue ceramic vase on the front counter, ready for a customer who ordered them. While Leah steps into the back room to get more ribbon, her colleague Sam decides to help by replacing the wilted roses with fresh ones, but he p
- **BigToM** — {"story": "Priya is a gardener at a community garden. She carefully fills a large watering can with water and places it next to the tomato plants, planning to water them after her lunch break. While Priya is inside eating, a sudden gust of wind knocks the watering can over, spilling all the water on

### Gaps
*(to be filled after manual review)*

### Next
*(to be filled after evaluation)*

---

## Iteration 4 — 2026-05-20 21:29
**Stage**: synth
**Datasets**: BigToM
**Synthesis model**: deepseek-v4-flash
**samples_per_batch**: 5 | **samples_per_report**: 3 | **max_retries**: 3

### Stats
- **BigToM**: bad_cases=? | reports=? | raw=21 | clean=21 | dropped=0 | hard=0
### Samples (auto)
- **BigToM** — {"story": "Ravi is a gardener at a community garden. He carefully fills a large blue watering can with water and places it next to the tomato plants, planning to water them in the afternoon. While Ravi is taking a lunch break inside the shed, a sudden gust of wind knocks the watering can over, spill
- **BigToM** — {"story": "Leah is a florist. She carefully arranges a dozen red roses in a blue ceramic vase on the front counter, ready for a customer who ordered them. While Leah steps into the back room to get more ribbon, her colleague Sam decides to help by replacing the wilted roses with fresh ones, but he p
- **BigToM** — {"story": "Priya is a gardener at a community garden. She carefully fills a large watering can with water and places it next to the tomato plants, planning to water them after her lunch break. While Priya is inside eating, a sudden gust of wind knocks the watering can over, spilling all the water on

### Gaps
*(to be filled after manual review)*

### Next
*(to be filled after evaluation)*

---

## Iteration 4 — 2026-05-20 21:36
**Stage**: difficulty
**Datasets**: BigToM
**Synthesis model**: deepseek-v4-flash
**samples_per_batch**: 5 | **samples_per_report**: 3 | **max_retries**: 3

### Stats
- **BigToM**: bad_cases=? | reports=? | raw=? | clean=? | dropped=? | hard=5
### Samples (auto)
- **BigToM** — {"story": "Ravi is a gardener at a community garden. He carefully fills a large blue watering can with water and places it next to the tomato plants, planning to water them in the afternoon. While Ravi is taking a lunch break inside the shed, a sudden gust of wind knocks the watering can over, spill
- **BigToM** — {"story": "Leah is a florist. She carefully arranges a dozen red roses in a blue ceramic vase on the front counter, ready for a customer who ordered them. While Leah steps into the back room to get more ribbon, her colleague Sam decides to help by replacing the wilted roses with fresh ones, but he p
- **BigToM** — {"story": "Priya is a gardener at a community garden. She carefully fills a large watering can with water and places it next to the tomato plants, planning to water them after her lunch break. While Priya is inside eating, a sudden gust of wind knocks the watering can over, spilling all the water on

### Gaps
*(to be filled after manual review)*

### Next
*(to be filled after evaluation)*

---

## Iteration 5 — 2026-05-20 22:02
**Stage**: all
**Datasets**: ToMBench, BigToM, SocialIQA, EmoBench, FanToM, HiToM, SimpleToM
**Synthesis model**: deepseek-v4-flash
**samples_per_batch**: 5 | **samples_per_report**: 3 | **max_retries**: 3

### Stats
- **ToMBench**: bad_cases=60 | reports=34 | raw=102 | clean=102 | dropped=0 | hard=31
- **BigToM**: bad_cases=60 | reports=13 | raw=39 | clean=39 | dropped=0 | hard=14
- **SocialIQA**: bad_cases=60 | reports=16 | raw=48 | clean=48 | dropped=0 | hard=7
- **EmoBench**: bad_cases=60 | reports=12 | raw=36 | clean=36 | dropped=0 | hard=16
- **FanToM**: bad_cases=60 | reports=14 | raw=42 | clean=42 | dropped=0 | hard=9
- **HiToM**: bad_cases=60 | reports=15 | raw=38 | clean=38 | dropped=0 | hard=37
- **SimpleToM**: bad_cases=60 | reports=13 | raw=39 | clean=39 | dropped=0 | hard=5
### Samples (auto)
- **ToMBench** — {"story": "在年终项目评审会上，林涛展示了他带领团队熬夜完成的季度报告。他的同事周敏注意到，当老板称赞报告的数据分析部分时，林涛短暂地瞥了一眼坐在角落的实习生小陈，嘴角微微上扬了一下。周敏知道，那份数据分析实际上是上周小陈私下给她看过的个人练习作品，林涛从未提过小陈的贡献。", "question": "周敏看到林涛瞥向小陈并微笑时，她最可能认为林涛的这个非言语动作意图是什么？A. 林涛在感谢小陈提供了数据，想用微笑表达认可。B. 林涛在确认小陈没有当众揭穿他窃取功劳的行为。C. 林涛在向小陈示意让他也参与讨论，以便分享荣誉。D. 林涛只是习惯性地扫视房间，微笑是因为项目成功了。",
- **ToMBench** — {"story": "在年度产品评审会上，陈经理展示了他主导的新款智能手表原型。同事李总监注意到手表的健康监测算法似乎直接复制了他团队去年未获批准的一个内部方案中的核心代码段。陈经理微笑着对李总监说：“你的团队之前做的市场调研真是帮了大忙，给了我很多启发。” 李总监回以微笑，但会后他悄悄把一份包含原始代码日期的文档副本发给了部门主管。", "question": "李总监为什么在会后把原始代码日期的文档副本发给部门主管？ A. 他真心感谢陈经理在评审会上的夸奖，想分享更多细节帮助陈经理改进产品。 B. 他想向部门主管证明陈经理的新款智能手表原型使用了李总监团队未获批准的方案，从而揭露抄袭行为。
- **ToMBench** — {"story": "In a competitive marketing department, Lina was vying with her colleague, Raj, for the coveted team lead position. Their manager, Mr. Chen, had asked everyone to submit peer feedback forms anonymously. Lina noticed Raj giving her a warm smile and a thumbs-up before the deadline, but she a

### Gaps
*(to be filled after manual review)*

### Next
*(to be filled after evaluation)*

---

## Iteration 6 — 2026-05-20 22:41
**Stage**: synth
**Datasets**: FanToM
**Synthesis model**: deepseek-v4-flash
**samples_per_batch**: 5 | **samples_per_report**: 5 | **max_retries**: 3

### Stats
- **FanToM**: bad_cases=? | reports=? | raw=510 | clean=510 | dropped=0 | hard=134
### Samples (auto)
- **FanToM** — {"story": "Liam, Chloe, and Ethan are planning a surprise party for their friend Mia. Liam says, \"Mia's favorite cake is red velvet, and she loves the bakery on Oak Street. Let's order from there.\" Chloe nods and says, \"Perfect. I'll call them tomorrow to place the order.\" Ethan adds, \"Great, I
- **FanToM** — {"story": "Chloe: I just heard from the lab that the prototype's battery is failing. We need a replacement by Friday.\nDaniel: I can call the supplier. What's the part number?\nChloe: It's B-772. They said it's the last one in stock.\nDaniel: Got it. I'll order it now.\n(Emma walks into the room.)\n
- **FanToM** — {"story": "Liam, Nora, and Owen are at a coffee shop. Liam says, \"I just heard that the art gallery downtown is closing permanently next month.\" Nora replies, \"Oh no, that's a shame. I love their Sunday brunch events.\" Owen nods. Just then, Chloe arrives and sits down. \"Sorry I'm late! What did

### Gaps
*(to be filled after manual review)*

### Next
*(to be filled after evaluation)*

---

## Iteration 6 — 2026-05-20 22:52
**Stage**: synth
**Datasets**: HiToM
**Synthesis model**: deepseek-v4-flash
**samples_per_batch**: 5 | **samples_per_report**: 5 | **max_retries**: 3

### Stats
- **HiToM**: bad_cases=? | reports=? | raw=456 | clean=456 | dropped=0 | hard=446
### Samples (auto)
- **HiToM** — {"story": "1. Clara places the key on the desk.\n2. David enters the room and sees the key on the desk.\n3. David moves the key from the desk to the drawer.\n4. Clara leaves the room.\n5. Emma enters the room and sees the key in the drawer.\n6. Emma moves the key from the drawer to the drawer (same 
- **HiToM** — {"story": "1. The living room contains a red sofa, a blue rug, and a wooden table. 2. The kitchen contains a silver fridge and a white counter. 3. The bedroom contains a brown dresser and a green chair. 4. The object is a small brass bell. 5. The bell is initially on the wooden table in the living r
- **HiToM** — {"story": "1. Liam places a key in the red drawer. 2. Emma is in the room and sees Liam place the key in the red drawer. 3. Emma leaves the room. 4. Liam moves the key from the red drawer to the blue drawer. 5. Emma re-enters the room. 6. Liam leaves the room. 7. Emma moves the key from the blue dra

### Gaps
*(to be filled after manual review)*

### Next
*(to be filled after evaluation)*

---

## Iteration 1 — 2026-05-21 00:06
**Stage**: diagnose
**Datasets**: ToMBench, BigToM, SocialIQA, EmoBench, FanToM, HiToM, SimpleToM
**Synthesis model**: deepseek-v4-flash
**samples_per_batch**: 5 | **samples_per_report**: 2 | **max_retries**: 3

### Stats
- **ToMBench**: bad_cases=? | reports=9 | raw=? | clean=? | dropped=? | hard=?
- **BigToM**: bad_cases=? | reports=9 | raw=? | clean=? | dropped=? | hard=?
- **SocialIQA**: bad_cases=? | reports=9 | raw=? | clean=? | dropped=? | hard=?
- **EmoBench**: bad_cases=? | reports=9 | raw=? | clean=? | dropped=? | hard=?
- **FanToM**: bad_cases=? | reports=8 | raw=? | clean=? | dropped=? | hard=?
- **HiToM**: bad_cases=? | reports=8 | raw=? | clean=? | dropped=? | hard=?
- **SimpleToM**: bad_cases=? | reports=8 | raw=? | clean=? | dropped=? | hard=?
### Samples (auto)
- **ToMBench** — {"story": "在年终项目评审会上，林涛展示了他带领团队熬夜完成的季度报告。他的同事周敏注意到，当老板称赞报告的数据分析部分时，林涛短暂地瞥了一眼坐在角落的实习生小陈，嘴角微微上扬了一下。周敏知道，那份数据分析实际上是上周小陈私下给她看过的个人练习作品，林涛从未提过小陈的贡献。", "question": "周敏看到林涛瞥向小陈并微笑时，她最可能认为林涛的这个非言语动作意图是什么？A. 林涛在感谢小陈提供了数据，想用微笑表达认可。B. 林涛在确认小陈没有当众揭穿他窃取功劳的行为。C. 林涛在向小陈示意让他也参与讨论，以便分享荣誉。D. 林涛只是习惯性地扫视房间，微笑是因为项目成功了。",
- **ToMBench** — {"story": "在年度产品评审会上，陈经理展示了他主导的新款智能手表原型。同事李总监注意到手表的健康监测算法似乎直接复制了他团队去年未获批准的一个内部方案中的核心代码段。陈经理微笑着对李总监说：“你的团队之前做的市场调研真是帮了大忙，给了我很多启发。” 李总监回以微笑，但会后他悄悄把一份包含原始代码日期的文档副本发给了部门主管。", "question": "李总监为什么在会后把原始代码日期的文档副本发给部门主管？ A. 他真心感谢陈经理在评审会上的夸奖，想分享更多细节帮助陈经理改进产品。 B. 他想向部门主管证明陈经理的新款智能手表原型使用了李总监团队未获批准的方案，从而揭露抄袭行为。
- **ToMBench** — {"story": "In a competitive marketing department, Lina was vying with her colleague, Raj, for the coveted team lead position. Their manager, Mr. Chen, had asked everyone to submit peer feedback forms anonymously. Lina noticed Raj giving her a warm smile and a thumbs-up before the deadline, but she a

### Gaps
*(to be filled after manual review)*

### Next
*(to be filled after evaluation)*

---

## Iteration 1 — 2026-05-21 00:23
**Stage**: synth
**Datasets**: ToMBench, BigToM, SocialIQA, EmoBench, FanToM, HiToM, SimpleToM
**Synthesis model**: deepseek-v4-flash
**samples_per_batch**: 5 | **samples_per_report**: 2 | **max_retries**: 3

### Stats
- **ToMBench**: bad_cases=? | reports=? | raw=45 | clean=45 | dropped=0 | hard=9
- **BigToM**: bad_cases=? | reports=? | raw=45 | clean=45 | dropped=0 | hard=9
- **SocialIQA**: bad_cases=? | reports=? | raw=45 | clean=45 | dropped=0 | hard=4
- **EmoBench**: bad_cases=? | reports=? | raw=45 | clean=45 | dropped=0 | hard=11
- **FanToM**: bad_cases=? | reports=? | raw=48 | clean=48 | dropped=0 | hard=15
- **HiToM**: bad_cases=? | reports=? | raw=43 | clean=43 | dropped=0 | hard=41
- **SimpleToM**: bad_cases=? | reports=? | raw=48 | clean=48 | dropped=0 | hard=0
### Samples (auto)
- **ToMBench** — {"story": "Tara and her friend Liam were both finalists in the school poetry slam, but Liam had to leave early to catch a flight and missed the final results. The next day, Tara saw Liam in the hallway and said, \"It's a shame you had to leave early—you would have loved the winning poem. Mine was pr
- **ToMBench** — {"story": "Liam and Chloe are both in the school's photography club. Last week, the club held a contest for the best photo of a sunset, but Chloe had a dentist appointment that day and missed the meeting where the winners were announced. Today, Liam walks up to Chloe in the hallway and says, 'I saw 
- **ToMBench** — {"story": "Tara and Jake are both in the school photography club. Last week, the club held a photo contest, and the results were announced on Friday. Tara was sick that day and missed the announcement. On Monday, Tara sees Jake in the hallway and says, 'I heard the judges really liked the way you ca

### Gaps
*(to be filled after manual review)*

### Next
*(to be filled after evaluation)*

---
