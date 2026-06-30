# 去冗余清理 Backlog

> **用途**：任务原则「每 10 个任务建立一个新任务去除冗余文件和代码」的取项清单。
> 每完成 10 个功能任务，从这里挑一项执行：消除重复实现、复用已有抽象、删除一次性文件。
> 每项标注：位置 / 现状 / 重构方向 / 优先级 / 风险。

执行规则：
- 一次只做一项，做完跑相关流水线验证（至少一个数据集端到端 / 对应 stage），再勾掉。
- 重构必须保持外部行为不变；不确定时先加最小测试固定当前输出。
- 勾掉后在本文件记录验证证据（跑了什么命令、结果）。

---

## P1 — 高价值，影响面大

### [ ] R1. `scripts/` 缺共享层，15 份 convert 样板重复
- **位置**：`scripts/convert_*.py`（15 个）
- **现状**：每个脚本都重复：①`OUT = Path(__file__).resolve().parent... / "datasets" / ...` 路径计算；②`main()` 落盘三件套（`mkdir` + `pd.DataFrame(columns=[story,question,answer,meta])` + `to_parquet` + 行数打印）；③`{"story":…, "answer":{"correct_answers":…,"wrong_answers":…}, "meta":…}` 行字面量；④结尾 `Counter` 分布打印。`scripts/` 下无 `__init__.py`，无任何共享工具。
- **重构方向**：新建 `scripts/_common.py`，抽出 `out_path(dataset, split)`、`make_row(story, question, correct, wrong, meta)`、`write_standard_parquet(rows, out_path, dist_fields=[...])`（含分布打印）。各 convert 脚本改为 import 复用。`convert_socialmind.py` 已有内部 `row()`/`make_meta()`，正好上移为公共实现。
- **风险**：低。convert 脚本是离线一次性运行，输出 parquet 可逐字节比对验证（重构前后对同一原始数据跑，diff parquet）。
- **验证**：选 2-3 个数据集重跑 convert，比对新旧 parquet 内容一致。

### [ ] R2. LLM 客户端加载两套并存
- **位置**：`filter/base.py`（`load_answer_models`/`load_judge_client`，写死读 `filter/config.yaml` + 手工字段映射）vs `src/runner.py`（`create_llm_client`/`create_content_client`，走 `Client.from_config(dict)`，配置源任意）
- **现状**：feedback 全程用 `runner.create_*`，filter 全程用 `filter.base`，两者最终构造相同的 `ContentClient`/`StructureClient`，只是配置来源与默认 `max_tokens` 不同。
- **重构方向**：filter 侧改为「自己读 yaml → 调 `runner.create_content_client/create_llm_client`」，删除 `_build_client` 的手工字段映射，只保留 strong/simple 角色到配置段的映射逻辑。
- **风险**：中。要确认 filter 的默认 `max_tokens`（answer 512 / judge 4096 / enable_thinking=False）在迁移后保持一致 —— `from_config` 的默认值与 filter 手工设的可能不同，需显式传入。
- **验证**：跑一个小数据集的 `run_filter.py`（max_samples 设小），确认 pass@k / answerability / repair 都能正常调用模型。

---

## P2 — 中等价值，局部冗余

### [ ] R3. parquet IO 三处各写各的
- **位置**：`filter/utils.write_parquet`（统一封装）vs `feedback/stage4_lsh_filter.save_as_parquet`（pyarrow，剔除 `_` 前缀字段）vs `filter/pipeline.py` 多处直接 `df.to_parquet(...)`（`load_split_df`/`labels_df`/`run_finalize`，未走自己的 `write_parquet`）
- **重构方向**：①`filter/pipeline.py` 内所有 `df.to_parquet` 改走 `filter.utils.write_parquet`（统一日志）；②评估 `save_as_parquet`（含「剔除 `_` 前缀字段」语义）是否能合并进 `write_parquet` 作为可选参数。读侧 filter 用 pandas、feedback 用 pyarrow 可暂不统一（影响小）。
- **风险**：低。
- **验证**：filter 端到端跑一遍，确认中间 parquet 正常生成。

### [ ] R4. summary 生成器归并循环重复
- **位置**：`report/generate_summary.py` 与 `report/generate_summary_html.py` 的 `generate_summary_table` 前半段（`collect_metrics_from_tables` → 遍历 `split_model_judge` → 填 `grouped` 字典）约 20 行逐字相同，只有渲染部分不同。
- **重构方向**：抽 `report/utils/summary_tables.py` 里新增 `build_summary_grid(tables_dir) -> (datasets, models, judges, grouped)`，两个生成器都调它，各自只保留渲染。
- **风险**：低。
- **验证**：`python report/generate_summary.py` + `generate_summary_html.py`，比对生成的 SUMMARY.md / html 与重构前一致。

### [ ] R5. 两套 Markdown 表格解析
- **位置**：`report/utils/markdown_tables.py`（`parse_markdown_table`/`parse_markdown_sections`，被 `generate_dataset_tables` 用）vs `generate_html_report.py` 自带的 `parse_md_table`（孤立实现，完全没 import `report.utils`）
- **重构方向**：把 `generate_html_report.py` 的解析统一到 `report.utils.markdown_tables`，必要时给 utils 版加一个「带 section_title」的返回变体。
- **风险**：中。两者返回结构不同（tuple vs dict），需小心适配 html 报告的渲染逻辑。
- **验证**：`python report/generate_html_report.py`，人工核对 html 表格与热力图渲染正常。

### [ ] R6. exp 目录解析 / 格式化函数重复
- **位置**：`generate_dataset_tables.resolve_exp_dir`（raise）vs `utils/results.find_experiment_dir`（返回 None）；`generate_dataset_tables.format_value` vs `utils/common.format_metric_value`
- **重构方向**：统一到 `utils` 版，调用方按需处理 None/异常；`format_value` 的 dict→json 分支合并进 `format_metric_value`。
- **风险**：低。
- **验证**：`python report/generate_dataset_tables.py` 输出与重构前一致。

---

## P3 — 低价值 / 设计约束 / 暂缓

### [ ] R7. feedback Stage1 判分与 filter 判分分叉
- **位置**：`feedback/stage1` 的 `_extract_letter`+`_is_correct`（正则取首字母，无 metrics.json 时 fallback）vs `filter/utils.is_correct_mcq/is_correct_open`
- **现状**：语义都是「判答案对错」，但 feedback 这套更粗糙、仅用于 bad case 召回 fallback。
- **重构方向**：评估能否让 stage1 fallback 复用 `filter.utils.is_correct_mcq`（注意循环依赖：feedback ← filter ← feedback.prompts，需理清）。
- **风险**：中（依赖方向）。**暂缓**，价值有限。

### [ ] R8. stage4 内 MinHash/Jaccard 循环局部重复
- **位置**：`feedback/stage4_lsh_filter.py` 的 `filter_candidates` 与 `deduplicate_internal` 各写一遍「`_make_minhash` → query → 精确 Jaccard」
- **重构方向**：抽一个公共比较函数。单文件内局部冗余，**优先级低**。

### [ ] R9. SocialMind 维度键编解码两端约定脆弱
- **位置**：`get_dimension_key`（stage1，编码 `dim__qtype__zh`）vs `_socialmind_parse_dim`（prompts，stage3 解码），靠注释维系
- **重构方向**：抽共享常量/编解码函数对（`encode_socialmind_dim`/`decode_socialmind_dim`）放一处。**优先级低**。

### [ ] R10. 18 份雷同 `run.py`
- **位置**：`tasks/*/run.py`
- **现状**：18 个逐字相同（仅转发到 `run_standardized_qa_task`），是 `run_eval` 子进程按路径约定调用的设计约束。
- **结论**：**保留**。低收益（每份仅 4 行），且物理入口文件是架构需要。除非改造 `run_eval` 的调用方式，否则不动。

### [ ] R11. 日志 basicConfig 多处调用
- **位置**：`src/llm/client.py`、`feedback/__init__.py`、`run_feedback.py`、`run_filter.py._setup_logging`
- **现状**：多次 `logging.basicConfig`，filter 侧清理 handlers、feedback 侧没有，多次 import 时以最先生效者为准。
- **重构方向**：收敛到一处日志初始化 helper。**优先级低**，影响仅日志格式。

---

## 一次性 / 待确认删除文件

执行清理任务时核查以下文件是否仍需保留（确认无引用后删除）：

- `scripts/restore_chinese_data.py`、`scripts/v4p2_apply_review_flags.py`、`scripts/v4p2_review_filter_compare.py` —— 一次性运维脚本，非转换样板。
- `report/v4p2_review_filter/`、`report/更新汇总_*.md` —— 历史快照/汇报文档，确认是否归档。
- `feedback/OVERVIEW.html`、`filter/OVERVIEW.html` —— 若由 `.md` 生成，可只留 `.md`。
- 根目录 `.DS_Store`、各级 `__pycache__/`、`.pytest_cache/` —— 应在 `.gitignore` 中，核查是否误入版本控制。

---

## 进度记录

| 项 | 状态 | 完成日期 | 验证证据 |
|---|---|---|---|
| R1 | 未开始 | | |
| R2 | 未开始 | | |
| R3 | 未开始 | | |
| R4 | 未开始 | | |
| R5 | 未开始 | | |
| R6 | 未开始 | | |
| R7~R11 | 暂缓/保留 | | |
