# feedback_synthesis 使用说明


当前入口是：

```bash
python run_feedback_synthesis.py
```

当前主配置文件是：

```text
feedback_synthesis/config.yaml
```

---

## 1. 核心思路

   1. 先从模型真实评测结果里找出模型做错的题
   2. 再分析这些错题背后的结构化错误模式
   3. 再围绕这些错误模式定向生成新样本
   4. 最后用泄漏过滤和难度过滤保留更有训练价值的数据
   5. 并把高价值样本反哺到下一轮合成

输入来源通常是评测阶段已经产出的：

```text
results/<dataset>/<model>/exp_*/prediction.jsonl
```

最终合成用于训练的结果是：

```text
data_output/synth_clean/<dataset>/synthetic_iter<N>_*_hard.parquet
```

---

## 2. 目录与入口

### 2.1 运行入口

整个合成链路的运行入口在仓库根目录：

```bash
python run_feedback_synthesis.py
```

### 2.2 主配置文件

主要配置文件在：

```text
feedback_synthesis/config.yaml
```

### 2.3 阶段实现文件

这条链路按 5 个阶段组织，主要文件如下：

```text
feedback_synthesis/
├── config.yaml
├── prompts.py
├── stage1_load_predictions.py
├── stage2_diagnosis.py
├── stage3_synthesis.py
├── stage4_lsh_filter.py
├── stage5_difficulty_filter.py
└── ITERATION_LOG.md
```

---

## 3. 运行前准备

### 3.1 config配置

当前流程至少会用到两类模型：

1. `synthesis_model`
   - 用于 Stage 2 的诊断
   - 用于 Stage 3 的合成
   - 也作为 Stage 5 的强模型复核模型

2. `difficulty_filter`
   - 用于 Stage 5 的弱模型过滤

因此需要检查 `feedback_synthesis/config.yaml` 中这两处的：

- `api_url`
- `api_key`
- `model_name`

除此之外还需要注意：
- `predictions_root` 控制 Stage 1 去哪里读评测结果，放prediction.jsonl
- `output_path` 控制整个合成链路的输出目录
- `leakage_guard.test_root` 控制 Stage 4 去哪里读取测试集做泄漏过滤
- `total_samples_budget` 控制本轮合成多少条数据（这里只是初始stage3合成数量，不代表最终留存数量）

### 3.2 确保已有评测结果

Stage 1 不是直接从原始数据集读取，而是从评测结果读取 bad case。  
所以需要确保 `results/` 下已经有对应模型的预测结果，例如：

```text
results/ToMBench/HardToM_GRPO_synthetic_20260515_hf/exp_20260515_120000/prediction.jsonl
```
### 3.3 确保已有测试集

stage4需要使用测试集去重，需要提前将测试集下载下来，放到指定位置，位置由config文件中leakage_guard.test_root确定。

---

## 4. 运行命令

### 4.1 跑完整流程

可以选择一个数据集，完整跑通一轮：

```bash
python run_feedback_synthesis.py --stage all --dataset ToMBench --iteration 1
```
iteration只代表迭代次数，影响生成文件的命名，确保每次不同即可，否则会覆盖已有数据。
这条命令会依次执行：

1. Stage 1：加载 bad case
2. Stage 2：维度诊断
3. Stage 3：样本合成
4. Stage 4：LSH 泄漏过滤
5. Stage 5：难度与有效性过滤

### 4.2 分阶段运行

也可以按阶段单独跑，例如生成report后可以只根据现有的report进行多次合成，或中途中断重启可以从当前阶段继续。

#### 只跑 Stage 1

```bash
python run_feedback_synthesis.py --stage load --dataset ToMBench --max-bad-cases 80
```

#### 只跑 Stage 2

```bash
python run_feedback_synthesis.py --stage diagnose --dataset ToMBench
```

#### 从 Stage 3 开始继续跑

```bash
python run_feedback_synthesis.py --stage synth --dataset ToMBench --iteration 1
```

注意：当前实现中，`--stage synth` **不只是 Stage 3**，而是：

```text
Stage 3 -> Stage 4 -> Stage 5
```

#### 只跑 Stage 4 + Stage 5

```bash
python run_feedback_synthesis.py --stage dedupe --dataset ToMBench --iteration 1
```

注意：当前实现中，`--stage dedupe` 会执行：

```text
Stage 4 -> Stage 5
```

#### 只跑 Stage 5

```bash
python run_feedback_synthesis.py --stage difficulty --dataset ToMBench --iteration 1
```

### 4.3 从 Stage 5 反馈报告继续合成（非必需）

bad case数据合成过程中会采集高价值样本报告，高价值样本：弱模型全错而强模型答对的样本。高价值样本超过阈值的报告会被提取保存到`data_output\diagnosis_reports\{DataSets}\stage5_feedback_iter{N}`

如果你已经完成了上一轮 Stage 5，并且已经得到高价值报告，根据需要下一轮可以直接从反馈报告继续合成，这里根据高价值报告的数量和合成情况自选，非必需。

```bash
python run_feedback_synthesis.py --stage synth --report-source stage5 --dataset ToMBench --iteration 2
```

这里的含义是：

- 本轮 `iteration=2`
- Stage 3 会读取上一轮的
  `stage5_feedback_iter1/dimension_reports.jsonl`

### 4.4 当前参数与联动关系

当前入口支持这些常用参数：

- `--stage`
- `--config`
- `--dataset`
- `--max-bad-cases`
- `--iteration`
- `--report-source`

一般修改`iteration`,`stage`,`report-source`即可

当前代码里的阶段联动关系是：

- `all` = 1 -> 2 -> 3 -> 4 -> 5
- `load` = 只跑 1
- `diagnose` = 只跑 2
- `synth` = 3 -> 4 -> 5
- `dedupe` = 4 -> 5
- `difficulty` = 只跑 5

---

## 5. 五个阶段分别做什么

### 5.1 Stage 1：加载 bad case

#### 输入

```text
results/<dataset>/<model>/exp_*/prediction.jsonl
```

#### 处理

Stage 1 会把多个模型的预测结果按 `sample_idx` 对齐，并判断哪些样本属于 bad case。  
当前规则仍然是：

- 只要任意模型在任意一次 repeat 中出错，该样本就进入 bad case 池

维度内部引入了 `error_score` 做优先级排序。

#### 输出

```text
data_output/bad_cases/<dataset>/bad_cases.jsonl
```

### 5.2 Stage 2：维度诊断

#### 输入

```text
data_output/bad_cases/<dataset>/bad_cases.jsonl
```

#### 处理

Stage 2 会先按维度对 bad case 分组，再让合成模型对同一维度的一批错题做批量诊断，抽象出：

- 共同错误模式
- 主要认知操作
- 推荐合成主题
- 难度分布

#### 输出

核心输出：

```text
data_output/diagnosis_reports/<dataset>/<subdir>/dimension_reports.jsonl
```

辅助输出：

```text
data_output/diagnosis_reports/<dataset>/<subdir>/dimension_coverage.json
```

### 5.3 Stage 3：从 report 合成新样本

#### 输入

默认读取：

```text
data_output/diagnosis_reports/<dataset>/.../dimension_reports.jsonl
```

如果使用 `--report-source stage5`，则读取上一轮反馈报告：

```text
data_output/diagnosis_reports/<dataset>/stage5_feedback_iter<N-1>/dimension_reports.jsonl
```

#### 处理

Stage 3 会读取诊断报告，为每条 report 构造合成 prompt，再按 `samples_per_report` 为每个 report 生成若干候选样本。

当前输出统一转成结构化格式：

- `story`
- `question`
- `answer`
- `meta`
- `data_source`

#### 输出

```text
data_output/synth_raw/<dataset>/candidates_iter<N>_<model>.jsonl
```

### 5.4 Stage 4：LSH 泄漏过滤

#### 输入

```text
data_output/synth_raw/<dataset>/candidates_iter<N>_<model>.jsonl
```

#### 处理

Stage 4 会把候选样本与测试集进行相似度比对，过滤掉可能泄漏测试集的样本。  
当前已经支持测试集索引缓存，因此同一份测试集不需要每次都重新全量建索引。

#### 输出

保留样本：

```text
data_output/synth_clean/<dataset>/synthetic_iter<N>_<model>.parquet
```

被丢弃样本：

```text
data_output/synth_clean/<dataset>/synthetic_iter<N>_<model>_dropped.jsonl
```

过滤日志：

```text
data_output/synth_clean/<dataset>/synthetic_iter<N>_<model>_dedupe_log.jsonl
```

### 5.5 Stage 5：弱模型筛简单题 + 强模型复核坏题

#### 输入

```text
data_output/synth_clean/<dataset>/synthetic_iter<N>_<model>.parquet
```

#### 处理

Stage 5 当前已经不再只是“弱模型错了就保留”。  
它现在会分三类处理：

1. 弱模型全对 -> 丢弃，认为太简单
2. 弱模型有对有错 -> 保留
3. 弱模型全错 -> 进入强模型复核

如果强模型至少答对一次，则保留；如果强模型始终答不对，则认为题目本身可能有问题，记为 `dropped_invalid_question` 并丢弃。

#### 输出

最终保留样本：

```text
data_output/synth_clean/<dataset>/synthetic_iter<N>_<model>_hard.parquet
```

同时，Stage 5 还会在诊断目录下写出高价值反馈报告：

```text
data_output/diagnosis_reports/<dataset>/stage5_feedback_iter<N>/dimension_reports.jsonl
```

## 6. 输出文件怎么看

### 6.1 `bad_cases/`

```text
data_output/bad_cases/<dataset>/bad_cases.jsonl
```

这是 Stage 1 输出的错误样本池，是后续诊断的输入。

### 6.2 `diagnosis_reports/`

```text
data_output/diagnosis_reports/<dataset>/...
```

这里保存 Stage 2 的维度诊断报告，以及 Stage 5 反馈报告。一般有：

- `dimension_reports.jsonl`
- `dimension_coverage.json`
- `stage5_feedback_iter<N>/dimension_reports.jsonl`

### 6.3 `synth_raw/`

```text
data_output/synth_raw/<dataset>/candidates_iter<N>_<model>.jsonl
```

这是 Stage 3 刚生成出来的候选样本，还没有经过泄漏过滤和难度过滤。

### 6.4 `synth_clean/*.parquet`

```text
data_output/synth_clean/<dataset>/synthetic_iter<N>_<model>.parquet
```

这是 Stage 4 通过泄漏过滤后的干净样本。  

### 6.5 `*_hard.parquet`

```text
data_output/synth_clean/<dataset>/synthetic_iter<N>_<model>_hard.parquet
```

这是 Stage 5 之后最终保留下来的样本，也是当前真正用来训练的结果。

### 6.6 `feedback_synthesis/ITERATION_LOG.md`

```text
feedback_synthesis/ITERATION_LOG.md
```

这里会记录每轮合成的摘要，包括：

- 当前 stage
- 数据集
- 合成模型
- bad case 数量
- 诊断 report 数量
- raw 样本数
- clean 样本数
- hard 样本数

---

## 7. 其他的一些概念


### 高价值样本

1. 弱模型全错
2. 强模型至少答对一次

这类样本对训练更有价值，因为它们更能体现“强弱模型之间的能力差距”。

### 高价值反馈报告

如果某个 report 生成出的样本里，高价值样本占比达到阈值，那么这份 report 会被重新写出为反馈报告。

因此，反馈报告并不是“单条高价值样本”，而是“更容易生成高价值样本的 report”。

### 6.5 如何利用高价值反馈报告

到了下一轮，你可以让 Stage 3 不再读取原始诊断报告，而是改为读取上一轮 Stage 5 的反馈报告。注意，如果要使用反馈报告，你需要保证：
 - 当前轮次没有已生成的数据（否则会覆盖已有数据）
 - 只会从上一轮次获取高价值反馈报告
 - 若要使用高价值反馈报告，stage只能选第三步合成，即synth。理解：从已有报告里合成数据。

```bash
python run_feedback_synthesis.py --stage synth --report-source stage5 --iteration N
```

