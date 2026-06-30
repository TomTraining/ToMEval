# 进度日志

## 当前已验证状态

- 仓库根目录：`/Users/yangmeili/Downloads/Code/BenchEval`
- 标准启动路径：`source /Users/yangmeili/Downloads/Code/.venv/bin/activate`（Python 解释器：`/Users/yangmeili/Downloads/Code/.venv/bin/python`）
- 标准验证路径：单数据集 `python tasks/<DS>/run.py`；模块导入 smoke test `python -c "import src.evaluation, filter.pipeline, feedback.stage1_load_predictions"`
- 当前最高优先级未完成功能：把框架从 ToM 专用扩展到任意测试集的全自动迭代式数据合成（见 `docs/vision.md`）
- 当前 blocker：无

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
