#!/usr/bin/env python3
"""filter 入口脚本（V3 数据飞轮）。

用法：
  python run_eval.py

所有配置（数据集、模型、路径等）在 filter/config.yaml 中设置。
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


def _setup_logging() -> None:
    """配置日志：同时输出到控制台和文件。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"filter_{timestamp}.log"

    # 清除之前的 handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    logging.info(f"日志文件: {log_file}")


def main() -> int:
    from new_sub.ToMEval.filter.run_filter import load_filter_config, run_filter_loop_all_splits, run_finalize
    from new_sub.ToMEval.filter.report_summary import save_summary_report
    import yaml
    from pathlib import Path

    _setup_logging()

    try:
        config = load_filter_config()
    except Exception as e:
        logging.error(f"配置加载失败: {e}")
        return 2

    # 读取数据集列表
    cfg_path = Path("filter/config.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    datasets = cfg.get("datasets", [])
    if not datasets:
        logging.error("config.yaml 中未配置 datasets")
        return 2

    logging.info(f"开始评估 {len(datasets)} 个数据集: {datasets}")

    # 对每个数据集运行完整流程：filter → finalize
    for dataset in datasets:
        logging.info(f"\n{'='*60}")
        logging.info(f"处理数据集: {dataset}")
        logging.info(f"{'='*60}")

        # Phase 1: filter（按 split 独立迭代评估 + 修复）
        try:
            summary = run_filter_loop_all_splits(
                dataset=dataset, max_iter=config["max_iter"], config=config
            )
            logging.info(
                f"[{dataset}] filter 完成 — splits={summary['splits_run']}/{summary['total_splits']} "
                f"total_keep={summary['total_keep_pool']}"
            )
        except FileNotFoundError as e:
            logging.warning(f"[{dataset}] 跳过（找不到数据文件）: {e}")
            continue
        except Exception as e:
            import traceback
            logging.error(f"[{dataset}] filter 失败: {e}")
            logging.error(f"完整错误堆栈:\n{traceback.format_exc()}")
            continue

        # Phase 2: finalize（合并所有 split 的 hard+medium → train_set）
        try:
            out_path = run_finalize(dataset=dataset, config=config)
            logging.info(f"[{dataset}] finalize 完成 — train_set → {out_path}")
        except FileNotFoundError as e:
            logging.warning(f"[{dataset}] finalize 跳过（无评估结果）: {e}")
        except Exception as e:
            logging.error(f"[{dataset}] finalize 失败: {e}")

    logging.info(f"\n{'='*60}")
    logging.info("所有数据集评估完成")
    logging.info(f"{'='*60}")

    # 生成汇总报告
    try:
        report_path = f"{config['output_root']}/SUMMARY_REPORT.md"
        save_summary_report(config["output_root"], datasets, report_path)
        logging.info(f"汇总报告已生成: {report_path}")
    except Exception as e:
        logging.error(f"生成汇总报告失败: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
