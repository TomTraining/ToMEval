# 模块参考手册（代码复用索引）

> **用途**：执行任务前先查这里，确认「我要做的事」是否已有现成实现，避免重复造轮子。
> 本手册按 `src/` 共享核心 → `tasks/` 数据集插件 → `filter/feedback` 两条流水线 → `report/scripts` 周边工具 的顺序组织，每个模块标注「职责 / 关键函数 / 复用点 / import 路径」。

整体架构是「**薄插件 + 厚共享核心**」：数据集差异全部下沉到 `tasks/<DS>/` 的三个扩展点（`config.yaml` / `metrics.py` / `prompt.py`），`src/` 核心代码对数据集完全无关。判分（rule / f1 / llm_simple / rubric）与协议（direct / cot / del_tom / direct_think）均为**注册表式可插拔**设计。

核心数据流：

```
原始异构数据
  └ scripts/convert_*.py        标准化 → datasets/<DS>/*.parquet（schema: story/question/answer/meta）
        └ run_eval.py           批量编排 → 子进程调 tasks/<DS>/run.py
              └ src/evaluation/pipeline.py   predict → metric → visualize
                    └ results/<DS>/<model>/exp_*/{prediction.jsonl, metrics.json}
                          └ report/generate_*.py   聚合 → tables/
                          └ feedback/              bad case → 合成新训练数据
                          └ filter/                训练数据质量评估 + 修复
```

标准 schema（贯穿全仓库）：

```json
{
  "story": "背景故事",
  "question": "问题",
  "answer": {"correct_answers": ["..."], "wrong_answers": ["...", "..."]},
  "meta": {"id": "...", "其他分组字段": "..."}
}
```

题型由 `answer` 自动推断：无 `wrong_answers` 且唯一 `correct` → `open`；多个 `correct` → `mcq_multi`；否则 `mcq_single`；`mcq_grouped` 由数据集 `prepare_samples` 钩子预打包。

---

## 一、`src/` —— 数据集无关的共享核心

### src/runner.py —— 配置加载 + 客户端工厂
- `load_experiment_config(config_path)`：读 `experiment_config.yaml`；含 `protocol` 时用协议覆盖采样参数、把 `repeats` 改成协议 `n_samples`。
- `create_content_client(llm_config) -> ContentClient`：文本生成客户端（predict 用）。
- `create_llm_client(llm_config) -> StructureClient`：结构化客户端（judge 用）。
- `load_and_limit_data(subset, datasets_root, max_samples, seed=42)`：取数 + 固定种子限量。

**复用点**：任何新流程要 judge/生成客户端，直接 `from src import runner; runner.create_llm_client(cfg)`，配置源可以是任意 dict（不限于 experiment_config）。`open_judge.build_open_ctx`、`feedback` 全程都复用它。

### src/dataloader/ —— 取数
- `load_dataset(subset, datasets_root=None) -> List[Dict]`：加载单个子集（arrow 目录或 parquet），找不到抛 `FileNotFoundError`。
- `list_subsets(datasets_root=None)`、`DataLoader.load_all(prefix)`：批量合并多子集。

**复用点**：`from src.dataloader import load_dataset` —— 通用取数入口，新数据只要落成 parquet/arrow 即可加载，无需改代码。

### src/evaluation/ —— 评测核心

| 模块 | 职责 | 高频复用点 |
|---|---|---|
| `pipeline.py` | 三阶段编排总入口（predict/metric/visualize/all） | `run_standardized_qa_task(config_path)` 是 CLI 主入口；三个 stage 函数可单独调 |
| `types.py` | TypedDict 契约 | `StandardizedSample / PredictionRecord / AnswerBlock / PromptType` |
| `data.py` | 标准化样本加载与校验 | `load_task_config`、`normalize_sample`、`analyze_question_types`、`read_yaml` |
| `prompts.py` | prompt 构造 + 选项装配 + `\boxed{}` 抽取 | `build_prompt`、`build_option_bundle`、`prompt_type`、`boxed_directive`、`render_options_block`、`extract_prediction_from_text`、`extract_boxed`、`OPEN_QA_TEMPLATE(_ZH)`、`CHOICE_QA_TEMPLATE(_ZH)` |
| `protocols.py` | 协议注册表（采样参数/system prompt/extractor） | `system_prompt_for`、`sampling_params_for_v4`、`extract_direct/extract_cot`、`is_voting_protocol`、`reasoning_for` |
| `prompt_loaders.py` | 动态加载 `tasks/<DS>/prompt.py` 钩子 | `load_task_prompt_builder / load_task_prepare_samples / load_task_system_prompt_builder` |
| `prediction.py` | 批量预测核心 | `predict_records(samples, dataset_name, client, repeats, protocol)` |
| `judge.py` | 判分分派（MCQ 规则 + open 委托） | `judge_repeat`（主入口）、`rule_judge_mcq`、`rule_judge_grouped`、`judge_prompt`、`backfill_meta` |
| `open_judge.py` | open 题判分注册表（f1/llm_simple/rubric） | `build_open_ctx`、`judge_open`、`max_f1(pred, golds, lang)`、`OpenJudgeContext`、`RubricResult` |
| `judge_schema.py` | judge 输出 schema | `QAJudgeResult(is_correct: bool)` |
| `lang.py` | 语言归一 | `get_sample_lang(meta) -> "zh"\|"en"` |
| `metrics.py` | 动态加载并执行数据集指标函数 | `aggregate_metrics(dataset_name, records, per_sample_results)` |
| `task_metrics.py` | 数据集 `metrics.py` 的共享积木 | `generic_group_metrics`、`by_meta`、`base_metric_payload`、`group_all_correct`、`safe_div`、`update_group/rate_dict/count_dict` |
| `storage.py` | 预测/指标读写 + LLMResponse 序列化 | `pred_content`、`serialize_llm_response`、`write/read_prediction_file`、`save_metrics` |
| `paths.py` | 实验目录管理 + config 脱敏落盘 + 跨 repeat 平均 | `build_experiment_dir`、`find_latest_experiment_dir`、`save_config`、`average_metrics` |
| `voting.py` | del_tom 多数投票折叠 | `vote_collapse`、`majority_vote_letter` |

**两条隐式契约（重要）**：
1. `task_metrics.generic_group_metrics` 产出的 `by_*`（准确率）/ `*_counts` 命名约定，正是 `visualization` 自动出图的依据 —— 写新数据集指标用 `generic_group_metrics + by_meta` 是最快路径，且免费获得图表。
2. open 判分一切都走 `OpenJudgeContext`：judge / voting / pipeline 只认这个上下文，新增 open 判分模式只改 `open_judge.OPEN_JUDGE_MODES` 注册表与分派。

### src/llm/ —— OpenAI 兼容客户端
- `client.py`：基类 `LLMClient`（`from_config`、`_build_messages`、线程安全 token 统计、`_split_think_content` 剥离 `<think>`）；`LLMResponse(content, reasoning)` 是全流程响应契约；`LLMUsage`。
- `content_client.py`：`ContentClient.generate / batch_generate(prompts, system_prompts=None)` —— 线程池并行，支持 per-sample system prompt。
- `structure_client.py`：`StructureClient.generate_structure / batch_generate_structure(mode="auto")` —— parse/create 双模式自动探测降级。
- `llm_utils.py`：`extract_json`（多策略 JSON 提取）、`format_schema_for_prompt`、`build_extra_body(top_k, enable_thinking)`（vLLM thinking 开关）。

**复用点**：`from src.llm import ContentClient, StructureClient, LLMResponse`。模型不支持原生结构化输出时 `StructureClient` 自动降级到「prompt 注入 schema + 正则提取」，无需手动干预。

### src/visualization/ —— 数据集无关出图
- `plots.py`：`render_single(payload, out_dir)`（单 payload 出全套图，pipeline visualize stage 调用）；`plot_model_comparison`（多模型柱状/雷达）；`load_metrics / primary_metrics`；单图函数 `plot_group_accuracy / plot_heatmap / plot_score_group / plot_judge_agreement`。
- `__main__.py`：CLI `python -m src.visualization --results <path...> --out <dir>`。

**复用点**：只消费 `metrics.json` 的 `by_*` 分组，完全数据集无关。`from src.visualization import render_single, plot_model_comparison`。

---

## 二、`tasks/<DS>/` —— 数据集插件层（薄）

每个数据集目录是一个薄插件，只声明差异：

| 文件 | 必需 | 作用 | 加载方式 |
|---|---|---|---|
| `run.py` | 是 | 物理入口，转发到 `run_standardized_qa_task("config.yaml")` | `run_eval.py` 子进程调用 |
| `config.yaml` | 是 | `dataset`/`path` 必填，可选 `open_judge`/`judge1`/`judge2`/`rubric`/`prompt_style` | `data.load_task_config` |
| `metrics.py` | 是 | `compute_metrics(records, per_sample_results)` | `metrics.load_task_metric_fn`（动态 import） |
| `prompt.py` | 否 | `build_prompt` / `build_system_prompt` / `prepare_samples` 钩子 | `prompt_loaders`（约定式加载，缺省回退通用实现） |

**关键事实**：
- 19 个数据集里 **18 个 `run.py` 逐字相同**（仅转发），是设计约束（`run_eval` 按路径约定子进程调用），低收益但属样板重复。
- `metrics.py` 复杂度谱：最简（HellaSwag/ToMBench，一行 `generic_group_metrics + by_meta`）→ 中（FanToM，加 set 级 ALL）→ 繁（BigToM 配对统计 / SocialMind 8 维 + Q4 rubric 均分）。
- 除 SocialMind 外都有 `prompt.py`（忠实复刻原论文题面）；只有 EmoBench 额外用 `prepare_samples`（EU 子集合并为 `mcq_grouped`）。
- 19 个数据集各自考察什么、从哪改造、如何判分，详见 [datasets.md](datasets.md)。

**新增数据集最快路径**（import 速查）：

| 需求 | 复用点 |
|---|---|
| 取数 | `from src.dataloader import load_dataset` |
| 标准化 | `from src.evaluation.data import normalize_sample` |
| 题型/选项 | `from src.evaluation.prompts import prompt_type, build_option_bundle` |
| 通用 prompt（缺省回退） | `from src.evaluation.prompts import build_prompt` |
| 指标积木 | `from src.evaluation.task_metrics import generic_group_metrics, by_meta` |
| open 判分 | `from src.evaluation.open_judge import build_open_ctx, max_f1` |
| MCQ 判分 | `from src.evaluation.judge import rule_judge_mcq` |

---

## 三、`filter/` —— 数据质量评估 + 自动修复流水线

入口 `run_filter.py` → `filter.pipeline`，最小单元是单个 split（一个 parquet）。决策树评估链路：pass@k 分桶 → answerability 判断 → shortcut 三维探测 → 打标 → 修复迭代 → finalize。

| 模块 | 职责 | 关键函数 |
|---|---|---|
| `base.py` | 按角色（strong/simple）构造客户端 | `load_answer_models`、`load_judge_client` |
| `utils.py` | 判分/IO/规范化基座 | `is_correct_open`（复用 `open_judge.max_f1`）、`is_correct_mcq`（复用 `prompts.extract_prediction_value`）、`write_parquet`、`resolve_sample_id`、`row_to_sample` |
| `prompts.py` | Phase C answerability prompt | `ANSWERABILITY_FULL_PROMPT(_ZH)` |
| `eval/eval_passk.py` | Phase B 难度分桶 | `run_passk_on_df(df, dataset, k, simple_client)` → bucket ∈ {all_passed, partial, all_failed} |
| `eval_answerability_full.py` | Phase C 可回答性 | `run_answerability_on_df` → label ∈ {answerable, label_error, ambiguous, contradictory_premise, missing_info} |
| `eval/eval_shortcut.py` | Phase D shortcut 探测 | `run_shortcut_on_df`；`build_no_story/no_question/no_options_prompt`；no_options 已改 F1 判分免 judge |
| `repair/repair_prompts.py` | Phase E 修复 prompt | `build_repair_prompt`、`get_dataset_focus`（复用 `feedback.prompts.DATASET_SKILL_REGISTRY`） |
| `repair/repair_pipeline.py` | Phase E 修复 | `repair_samples`（强 schema `RepairedSample` 批量生成 + `meta.history` 留痕） |
| `pipeline.py` | 决策树编排器（核心） | `run_filter_loop_all_splits`、`run_filter_loop`、`run_single_iteration`、`assign_labels`、`_assign_one`（决策树规则）、`normalize_df`（Arrow→list 类型规范化）、`run_finalize` |
| `report_summary.py` | 汇总报告 | `generate_summary_report`、`save_summary_report` |

**决策树规则（`_assign_one`）**：

| bucket | answerable | is_shortcut | label | repair_type |
|---|---|---|---|---|
| all_passed | — | — | easy | easy |
| all_failed | True | — | hard | （保留） |
| all_failed | False/None | — | bad | unanswerable |
| partial | ≠True | — | bad | unanswerable |
| partial | True | True | shortcut | shortcut |
| partial | True | False | medium | （保留） |

最终 `_KEEP_LABELS = {hard, medium}`；`{easy, shortcut, bad}` 进修复，迭代到 `max_iter` 仍需修复 → `unfixable`。每阶段带 parquet 断点续跑。

**复用点**：`run_passk_on_df` / `run_shortcut_on_df` 都支持依赖注入 client，可脱离 config 单独用；`build_no_*_prompt` 是现成的消融变体 prompt 构造器；`repair_samples` 是「labels 驱动选行 + 强 schema 生成 + history 留痕」完整模式。

---

## 四、`feedback/` —— 错题反馈 → 维度诊断 → 合成 → 去重

入口 `run_feedback.py`，stage ∈ {all, load, diagnose, synth, dedupe}。

| 阶段 | 模块 | 职责 | 关键函数 |
|---|---|---|---|
| 1 load | `stage1_load_predictions.py` | 多模型 bad case 并集 + 维度统计 | `load_bad_cases_from_predictions`、`get_dimension_key`（跨数据集维度归一权威，被 stage1/2/3 共用） |
| 2 diagnose | `stage2_diagnosis.py` | 按维度诊断错误模式 | `run_stage2_dimension_diagnosis`、`allocate_reports_by_dimension`（配额分配）、`sample_bad_cases_weighted`（加权采样）、`DimensionDiagnosisReport` schema |
| 3 synth | `stage3_synthesis.py` | 诊断报告 → 新样本 | `synthesize_from_reports`；`SYNTHESIS_SCHEMA_REGISTRY`（按数据集选 schema）；`SOCIALMIND_SCHEMA_BY_QTYPE`；各 `*QuestionFlat.to_storage_dict()` |
| 4 dedupe | `stage4_lsh_filter.py` | MinHash LSH 双重去重（测试集泄漏 + 内部重复） | `build_test_index`、`filter_candidates`、`deduplicate_internal`、`_make_minhash`、`save_as_parquet` |
| — | `prompts.py` | 注册表 + prompt 构造 | `SYNTHESIS_FORMAT_REGISTRY`、`DATASET_SKILL_REGISTRY`（filter 复用）、`build_batch_diagnosis_prompt`、`build_stage2_generation_from_report_prompt` |
| — | `report_summary.py` | 汇总 | `generate_summary_report`、`save_summary_report` |

**复用点**：
- `feedback/stage4_lsh_filter.py` 是项目**唯一的相似度去重实现**，需要做去重直接 `from feedback.stage4_lsh_filter import build_test_index, filter_candidates, deduplicate_internal`（依赖 `datasketch`、`pyarrow`）。
- `get_dimension_key` 是跨数据集维度字段映射的唯一权威（ToMBench→ability，BigToM→condition_type，HiToM→order_N，FanToM→question_type，SocialMind→`dim__qtype`，非英文追加 `__{lang}`）。
- `allocate_reports_by_dimension` / `sample_bad_cases_weighted` 是通用配额/加权采样算法。

---

## 五、`report/` —— 结果聚合与报告

数据流：`results/<DS>/<model>/exp_*/metrics.json` → `tables/<DS>/{基础指标,其他指标}.md` → `SUMMARY.md` / `report.html`。

| 文件 | 职责 |
|---|---|
| `generate_dataset_tables.py` | 每数据集 Markdown 表（最大，606 行，支持增量合并/多 exp 绑定） |
| `generate_summary.py` | 跨数据集 Markdown 总览 |
| `generate_summary_html.py` | 同上 HTML 合并单元格版 |
| `generate_html_report.py` | 单文件交互式 HTML（热力图 + tab） |
| `report_client.py` | bad case 分析报告（可选 LLM 解释） |
| `utils/` | 已抽好的共享层（`report.utils`） |

**复用点（`report.utils`）**：
- `common`：`load_yaml`、`read_json`、`parse_model_entry`、`format_metric_value`、`format_accuracy`
- `results`：`find_experiment_dir`、`load_metrics_payload`、`load_prediction_records`、`collect_result_bundles`、`iter_datasets`
- `markdown_tables`：`parse_markdown_table`、`parse_markdown_sections`
- `summary_tables`：`collect_metrics_from_tables`、`split_model_judge`、`config_display_name`、`parse_basic_metrics_table`
- `badcases`：`load_bad_cases`（分层抽样）

---

## 六、`scripts/convert_*.py` —— 标准化转换层

15 个 convert 脚本把原始异构数据转成统一 parquet schema，共享同一套样板（路径计算 → `build_rows()` → `pd.DataFrame(columns=[...])` → `to_parquet` → 分布打印）。`build_rows()` 是各脚本唯一真正不同的部分。

**注意**：`scripts/` 目前**没有任何共享工具模块**（无 `__init__.py`），样板代码 15 份重复 —— 这是任务 3 清理的重点目标（见 [refactor_backlog.md](refactor_backlog.md)）。`restore_chinese_data.py`、`v4p2_*.py` 是一次性运维脚本，不属转换样板。

---

## 七、跨模块复用速查表（最常用）

| 用途 | 复用点 | import 路径 |
|---|---|---|
| 取数 | `load_dataset` | `from src.dataloader import load_dataset` |
| 样本标准化 | `normalize_sample` | `from src.evaluation.data import normalize_sample` |
| 题型/选项装配 | `prompt_type`, `build_option_bundle` | `from src.evaluation.prompts import ...` |
| 通用 prompt | `build_prompt` | `from src.evaluation.prompts import build_prompt` |
| QA 模板 | `OPEN_QA_TEMPLATE(_ZH)`, `CHOICE_QA_TEMPLATE(_ZH)` | `from src.evaluation.prompts import ...` |
| 协议参数/system prompt | `system_prompt_for`, `sampling_params_for_v4` | `from src.evaluation.protocols import ...` |
| 指标积木 | `generic_group_metrics`, `by_meta`, `base_metric_payload`, `group_all_correct` | `from src.evaluation.task_metrics import ...` |
| open 判分 | `build_open_ctx`, `judge_open`, `max_f1` | `from src.evaluation.open_judge import ...` |
| MCQ 规则判分 | `rule_judge_mcq`, `judge_repeat` | `from src.evaluation.judge import ...` |
| 语言归一 | `get_sample_lang` | `from src.evaluation.lang import get_sample_lang` |
| LLM 客户端 | `ContentClient`, `StructureClient` | `from src.llm import ...` 或 `runner.create_*_client(cfg)` |
| JSON 提取 | `extract_json` | `from src.llm.llm_utils import extract_json` |
| parquet 写入 | `write_parquet` | `from filter.utils import write_parquet` |
| LSH 去重 | `build_test_index`, `filter_candidates`, `deduplicate_internal` | `from feedback.stage4_lsh_filter import ...` |
| 维度键归一 | `get_dimension_key` | `from feedback.stage1_load_predictions import get_dimension_key` |
| 出图 | `render_single`, `plot_model_comparison` | `from src.visualization import ...` |
| 报告共享层 | 见第五节 | `from report.utils import ...` |

---

## 八、已知冗余清单指针

代码库中已识别的重复实现与冗余点（client 加载两套、parquet IO 三处、判分逻辑分叉、convert 脚本无共享层等）已整理成可执行的清理 backlog，见 **[refactor_backlog.md](refactor_backlog.md)**。执行任务 3（每 10 个任务做一次去冗余）时从该文件取项。
