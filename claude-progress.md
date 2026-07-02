# 进度日志

## 当前已验证状态

- 仓库根目录：`/Users/yangmeili/Downloads/Code/BenchEval`
- 标准启动路径：`source /Users/yangmeili/Downloads/Code/.venv/bin/activate`（Python 解释器：`/Users/yangmeili/Downloads/Code/.venv/bin/python`）
- 标准验证路径：单数据集 `python tasks/<DS>/run.py`；模块导入 smoke test `python -c "import src.evaluation, filter.pipeline, feedback.stage1_load_predictions"`；端到端 smoke（eval+feedback+filter，ToMBench 小样本）`python scripts/run_smoke_pipeline.py`（需先在 .env 填 tokenkey 真实 key）
- 当前最高优先级未完成功能：把框架从 ToM 专用扩展到任意测试集的全自动迭代式数据合成（见 `docs/vision.md`）
- 当前 blocker：无
- smoke 测试流程：`python scripts/run_smoke_pipeline.py`（ToMBench 端到端 eval→feedback→filter，需先在 `.env` 填真实 key）

## 项目背景（一句话）

BenchEval = 给定任意测试集 + 训练 model + 强 teacher model API，自动化迭代式合成训练数据，经 SFT/RL 后显著提升模型在该测试集上的表现。三原则：流程自动化 / 轻量化 / 可视化。详见 `docs/vision.md`。

## 会话记录

### Session 001

- 日期：2026-06-30
- 本轮目标：(1) 把项目愿景写入 docs；(2) 通读代码库，把每个模块的用途/复用点写入 docs；(3) 建立冗余清理 backlog
- 已完成：
  - `docs/vision.md` —— 项目愿景与目标（任意数据集泛化、三原则、全自动迭代闭环）
  - `docs/module_reference.md` —— 模块级复用参考手册（src / tasks / report / scripts / filter / feedback 每个模块的职责、关键函数、import 路径、跨模块复用速查）
  - `docs/refactor_backlog.md` —— 冗余文件/重复代码清单，按优先级排序，作为"每10个任务一次清理"的待办池
  - README 文档索引补充三份新文档
  - 修正项目追踪文件（claude-progress.md / feature_list.json / init.sh）从 npm 模板改为本仓库实际情况
- 运行过的验证：派 3 个子代理全量精读 src/、tasks+report+scripts、filter+feedback（只读）
- 已记录证据：三份子代理精读报告已整合进 `docs/module_reference.md` 与 `docs/refactor_backlog.md`
- 提交记录：（待提交）
- 更新过的文件或工件：docs/vision.md、docs/module_reference.md、docs/refactor_backlog.md、README.md、claude-progress.md、feature_list.json、init.sh
- 已知风险或未解决问题：feature_list.json 里的"扩展到任意数据集"功能尚未拆成可验证的子任务；冗余清理尚未开始执行
- 下一步最佳动作：从 `docs/refactor_backlog.md` 选 P0 项（scripts/ 公共层抽取）开始第一轮清理，或按 vision.md 拆解"任意数据集泛化"的第一个可验证里程碑

### Session 002

- 日期：2026-07-01
- 本轮目标：按优先级推进 refactor backlog；先按用户要求终止不需要的项，再逐项执行
- 已完成：
  - 终止 R1 / R4 / R5：新增 `cancelled` 状态，三项标为 cancelled（用户决定后续整体清除，不再执行）
  - **R2** 收敛 LLM 客户端加载：`filter/base.py` 改为复用 `src.runner.create_content_client/create_llm_client`，删除手工 ContentClient/StructureClient 构造。关键：`_FILTER_DEFAULTS` 用 setdefault 注入 filter 旧默认（temp 0.3 / mw 8 / enable_thinking False / answer 512·judge 4096），config 显式字段优先 → 状态 in_progress（端到端 run_filter.py 缺输入数据+真实 key，待补测）
  - **R3** parquet 写入统一：`filter/pipeline.py` 5 处 `df.to_parquet` 全改走 `filter.utils.write_parquet`。评估结论：feedback.stage4.save_as_parquet(List[dict]+pyarrow+剔除_前缀) 与 filter 侧差异大，不合并 → 状态 in_progress（端到端待测）
  - **R6** 统一 exp 解析/格式化：`generate_dataset_tables` 删除本地 resolve_exp_dir/format_value，复用 `utils.results.find_experiment_dir` 与 `utils.common.format_metric_value`（后者增补 dict→json 分支）→ 状态 passing（等价性验证完整）
- 运行过的验证：
  - 模块 smoke test：`import src.evaluation, filter.pipeline, feedback.stage1_load_predictions, report.generate_dataset_tables` 通过
  - R2：构造 client 逐项核对类型/参数与旧手工实现一致
  - R3：write_parquet 行为等价（嵌套目录创建/index=False/空表可写）
  - R6：format 对 9 类输入、exp 解析对 3 种 happy-path 与旧实现逐项一致
- 已知风险或未解决问题：
  - R2/R3 端到端 run_filter.py 未跑（缺 `train_datasets/` 输入数据 + config.yaml 是占位 api_key），代码改动为行为等价、已单元级验证
  - R6 行为变化：显式指定但不存在的 exp_suffix，旧=抛异常列出可用实验，新=返回 None 静默跳过（符合 backlog『调用方按需处理 None/异常』）
- 下一步最佳动作：执行 refactor-cleanup（核查并清理一次性文件/历史快照/误入版本控制的缓存），每个文件删除前必须确认无引用

### Session 003

- 日期：2026-07-01
- 本轮目标：新建一个基于 qwen3-8b 的 ToMBench 端到端 smoke 测试流程（eval + feedback + filter），作为回归测试基线；可小样本，不跑全量
- 已完成（test-smoke-001，状态 passing —— 端到端已用真实 key 跑通）：
  - **filter 支持 `FILTER_CONFIG` 环境变量**：`filter/base.py` 的 `_CONFIG_PATH` 改为 `os.environ.get("FILTER_CONFIG") or filter/config.yaml`；pipeline.py 复用同一变量；run_filter.py 读 datasets 也改用 `base._CONFIG_PATH`。向后兼容（不设变量时仍读 filter/config.yaml）
  - **三份隔离 smoke 配置** `scripts/smoke/`：experiment_config.smoke.yaml（eval：qwen3-8b@tokenkey，datasets=[ToMBench]，max_samples=20，protocol cot）、feedback.smoke.yaml（teacher=deepseek-v4-flash@tokenkey，synthesis_datasets=[ToMBench]，models.name=qwen3-8b 对齐 eval，target=10）、filter.smoke.yaml（strong=deepseek / simple=qwen3-8b@tokenkey，datasets=[ToMBench]，max_samples=10，max_iter=1）。API key 用 `${VAR}` 占位符
  - **`.env` 占位符**（已被 .gitignore 忽略）：TOKENKEY_BASE_URL / TOKENKEY_QWEN_KEY / TOKENKEY_DEEPSEEK_KEY，先占位符，用时替换真实 key
  - **编排脚本 `scripts/run_smoke_pipeline.py`**：加载 .env → 展开 ${VAR} 写 .resolved 临时配置 → 依次跑 eval→feedback→[parquet 搬运]→filter，每段校验产物、任一失败即停；支持 --skip / --only；退出时清理 .resolved
  - 关键衔接发现：feedback 产物 `feedback_output/datasets/<DS>.parquet` 与 filter 输入 `train_datasets/<DS>/synthetic.parquet` 不自动衔接，编排脚本的 transfer 步骤负责搬运
- 运行过的验证：
  - 静态验证：py_compile 编排脚本通过；占位符检测正确拦截；${VAR} 展开与 .resolved 清理正常；三份 resolved 配置均可被下游加载器正确解析；ToMBench 数据可读 5720 行
  - **连通性预检**：修正 tokenkey base_url 为 `https://api.tokenkey.dev/v1`（不带 /v1 会返回官网 HTML）；qwen3-8b、deepseek-v4-flash 两个模型名 + key 均有效
  - **端到端真实跑通**（.env 填入 global 的 tokenkey key）：
    - eval：ToMBench 20 条 → results/ToMBench/qwen3-8b/exp_*/prediction.jsonl + metrics.json + 2 张图
    - feedback：20 bad case → 2 诊断报告 → 合成 10 条 → 去重后 10 clean → feedback_output/datasets/ToMBench.parquet
    - transfer：搬运到 train_datasets/ToMBench/synthetic.parquet
    - filter：pass@k(qwen3-8b) + answerability(deepseek) + shortcut + repair 全部真实调模型，2 轮迭代 → filter_output_smoke/datasets/ToMBench_filtered.parquet（3 行，可读）
  - 顺带补齐 R2/R3 端到端证据：filter 日志中 [passk]/[answerability]/[repair] 各角色 client 都正常调用（R2 收敛生效）；所有 parquet 都是 `wrote N rows →` 的 write_parquet 统一日志（R3 生效）→ R2/R3 一并转 passing
  - 认知修正：cot 协议 llm_overrides 实际为 max_tokens 32768 / enable_thinking True（早先探针记的 4096/False 有误）
- 本轮踩坑与修正：
  - tokenkey base_url 必须带 /v1，否则返回官网 HTML
  - qwen3-8b 非流式要求 enable_thinking=false，cot 协议是 True → smoke eval 配置改用 direct 协议（enable_thinking=False，且更快省 token，适合 smoke）
  - filter.smoke.yaml 输出到 filter_output_smoke/（隔离），但编排脚本原硬编码校验 filter_output/ → 已改为从 resolved 配置读 output_root
  - `scripts/` 整体被 .gitignore 忽略：已加例外 `!scripts/ !scripts/smoke/*.yaml` 等，让 smoke 脚本+配置可入库，但 .env 与含真实 key 的 .resolved 仍排除
  - feedback_output/ filter_output/ filter_output_smoke/ 加入 .gitignore（评测产物不入库）；本次临时产物已清理
- 已知风险或未解决问题：无（三段贯通已验证）
- 下一步最佳动作：执行 refactor-cleanup（核查并清理一次性文件/历史快照/误入版本控制的缓存），每个文件删除前确认无引用；或提交本轮改动

### Session 004

- 日期：2026-07-02
- 本轮目标：执行 refactor-cleanup（核查并清理一次性文件/历史快照/误入版本控制的缓存），每个文件删除前确认无引用
- 已完成（refactor-cleanup，状态 passing）：
  - **逐项核查 backlog「一次性/待确认删除文件」段的每个候选**，确认无代码引用后再处理：
    - `scripts/restore_chinese_data.py`、`scripts/v4p2_apply_review_flags.py`、`scripts/v4p2_review_filter_compare.py`：本轮核查时已不在磁盘（此前已删），无需处理
    - `feedback/OVERVIEW.html`、`filter/OVERVIEW.html`：已入库静态导览 HTML，全库无代码引用、无生成脚本 → `git rm`，保留 `.md` 源（可从历史恢复）
    - `report/更新汇总_0604-0611.md`、`report/更新汇总_0611.md`：已入库历史汇报文档，无代码引用 → `git rm`（可从历史恢复）
    - `report/v4p2_review_filter/`（3 文件：2 review md + summary.json）：**未入库且 gitignore 的历史快照**，全库无引用，删除不可恢复 → 经用户显式确认后 `rm -rf`
    - `.DS_Store`/`__pycache__/`/`.pytest_cache/`：`git ls-files` 核查无一被追踪，`.gitignore` 已覆盖（第 1/29/68/88/91 行），无误入版本控制
- 运行过的验证：
  - 删除前对每个候选跑 grep 全库引用扫描（排除文件自身与 .git）：OVERVIEW.html/更新汇总/v4p2_review_filter 仅被 feature_list.json 与 refactor_backlog.md 的清理清单自身提及，无代码引用
  - 删除后模块 import smoke test 通过：`import src.evaluation, filter.pipeline, feedback.stage1_load_predictions, report.generate_dataset_tables`
- 已知风险或未解决问题：无。tracked 文件均可从 git 历史恢复；untracked 快照经用户确认后删除
- 下一步最佳动作：提交本轮 refactor-cleanup 改动；后续可从 vision.md 拆解「任意数据集泛化」的第一个可验证里程碑
