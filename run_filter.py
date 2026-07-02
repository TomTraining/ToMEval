"""
数据过滤流水线入口

流程：
  1. pass@k 评估（全量强弱模型）
  2. answerability 判断
  3. shortcut 探测
  4. 修复迭代（unanswerable / shortcut → repair）
  5. finalize（合并 hard+medium → train_set）
  6. 生成汇总报告（SUMMARY_REPORT.md）

用法：
  python run_filter.py
  FILTER_CONFIG=path/to/config.yaml python run_filter.py   # 指定其他配置（如 smoke 测试）

配置说明：
  - 数据集、模型、路径等在 filter/config.yaml 中设置（可用 FILTER_CONFIG 环境变量覆盖路径）
  - 只跑部分数据集：在 config.yaml 的 datasets 列表里注释掉不需要的
"""

import logging
import sys
import yaml
from datetime import datetime
from pathlib import Path


def _setup_logging() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"filter_{timestamp}.log"
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
    from filter.pipeline import load_filter_config, run_filter_loop_all_splits, run_finalize
    from filter.report_summary import save_summary_report

    _setup_logging()

    try:
        config = load_filter_config()
    except Exception as e:
        logging.error(f"配置加载失败: {e}")
        return 2

    # 复用 base._CONFIG_PATH：默认 filter/config.yaml，可由 FILTER_CONFIG 环境变量覆盖
    from filter.base import _CONFIG_PATH as cfg_path
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    datasets = cfg.get("datasets", [])
    if not datasets:
        logging.error("config.yaml 中未配置 datasets")
        return 2

    logging.info(f"开始过滤 {len(datasets)} 个数据集: {datasets}")

    for dataset in datasets:
        logging.info(f"\n{'='*60}")
        logging.info(f"处理数据集: {dataset}")
        logging.info(f"{'='*60}")

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

        try:
            out_path = run_finalize(dataset=dataset, config=config)
            logging.info(f"[{dataset}] finalize 完成 — train_set → {out_path}")
        except FileNotFoundError as e:
            logging.warning(f"[{dataset}] finalize 跳过（无评估结果）: {e}")
        except Exception as e:
            logging.error(f"[{dataset}] finalize 失败: {e}")

    logging.info(f"\n{'='*60}")
    logging.info("所有数据集过滤完成")
    logging.info(f"{'='*60}")

    try:
        report_path = f"{config['output_root']}/SUMMARY_REPORT.md"
        save_summary_report(config["output_root"], datasets, report_path)
        logging.info(f"汇总报告已生成: {report_path}")
    except Exception as e:
        logging.error(f"生成汇总报告失败: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
