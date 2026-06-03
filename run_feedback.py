"""
数据合成流水线入口

流程：
  1. 从预测结果加载 bad case
  2. 维度批量诊断（stage2）
  3. 从诊断报告合成新样本（stage3）
  4. LSH 守门员过滤（stage4）
  5. 生成汇总报告（SUMMARY.md）

用法：
  python run_feedback.py                              # 使用默认配置
  python run_feedback.py --config feedback/config.yaml  # 指定配置文件

配置说明：
  - 只跑部分数据集：在 config.yaml 的 synthesis_datasets 列表里注释掉不需要的
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict

import yaml

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_pipeline(config_path: str):
    config = load_config(config_path)

    stage = config.get("stage", "all")
    max_bad_cases = config.get("max_bad_cases", 0)

    output_path = Path(config["output_path"])
    synthesis_llm_config = config["synthesis_model"]
    model_name = synthesis_llm_config.get("model_name", "unknown")

    if stage in ("all", "diagnose", "synth") and not synthesis_llm_config.get("api_key"):
        logger.error(
            "synthesis_model.api_key is empty in config.yaml. "
            "Please fill in your DashScope API key."
        )
        sys.exit(1)

    # ── 数据集列表 ────────────────────────────────────────────────────────────
    synthesis_datasets = config["synthesis_datasets"]
    if not synthesis_datasets:
        logger.error("No datasets configured in synthesis_datasets")
        sys.exit(1)

    dataset_names = [d["name"] for d in synthesis_datasets]
    logger.info(f"\n{'='*70}")
    logger.info(f"🚀 数据合成流水线启动")
    logger.info(f"{'='*70}")
    logger.info(f"  阶段: {stage}")
    logger.info(f"  数据集: {dataset_names}")
    logger.info(f"  Bad case 上限: {max_bad_cases if max_bad_cases > 0 else '无限制'}")

    syn_config = config.get("synthesis", {})
    max_retries = syn_config.get("max_retries_per_diagnosis", 3)
    bad_cases_per_report = syn_config.get("bad_cases_per_report", 1)
    diagnosis_batch_size = syn_config.get("diagnosis_batch_size", 10)
    synthesis_batch_size = syn_config.get("synthesis_batch_size", 20)
    predictions_root = config.get("predictions_root", "results")

    all_stats = {}

    # ── Stage 1: 加载 bad case ───────────────────────────────────────────────
    if stage in ("all", "load"):
        logger.info(f"\n{'='*70}")
        logger.info("📥 Stage 1: 加载模型答错的题目")
        logger.info(f"{'='*70}")
        from feedback.stage1_load_predictions import load_bad_cases_from_predictions

        for ds_info in synthesis_datasets:
            ds = ds_info["name"]
            models = ds_info.get("models", [])
            if not models:
                logger.warning(f"  ⚠️  [{ds}] 未配置模型，跳过")
                continue
            try:
                logger.info(f"  📂 [{ds}] 从 {len(models)} 个模型收集错题...")
                out_dir = load_bad_cases_from_predictions(
                    dataset_name=ds,
                    predictions_root=predictions_root,
                    models=models,
                    max_bad_cases=max_bad_cases,
                    output_dir=str(output_path / "_intermediate" / "bad_cases"),
                )
                bc_file = out_dir / "bad_cases.jsonl"
                with open(bc_file, encoding="utf-8") as f:
                    n = sum(1 for _ in f)
                all_stats.setdefault(ds, {})["bad_cases"] = n
                logger.info(f"  ✅ [{ds}] 收集到 {n} 道错题")
            except Exception as e:
                logger.error(f"  ❌ [{ds}] 加载失败: {e}")
                traceback.print_exc()

    # ── Stage 2: 维度诊断 ─────────────────────────────────────────────────────
    if stage in ("all", "diagnose"):
        logger.info(f"\n{'='*70}")
        logger.info("🔍 Stage 2: 分析错误模式（按能力维度诊断）")
        logger.info(f"{'='*70}")
        from feedback.stage2_diagnosis import run_stage2_dimension_diagnosis

        for ds_info in synthesis_datasets:
            ds = ds_info["name"]
            bad_cases_dir = output_path / "_intermediate" / "bad_cases" / ds
            if not bad_cases_dir.exists():
                logger.warning(f"  ⚠️  [{ds}] 未找到错题目录，跳过诊断")
                continue

            target_samples = ds_info.get("target_samples", 5000)
            samples_per_report = ds_info.get("samples_per_report", 5)
            total_reports = max(1, target_samples // samples_per_report)

            logger.info(
                f"  📊 [{ds}] 目标生成 {target_samples} 条数据 → 需要 {total_reports} 份诊断报告"
                f"（每报告输入 {bad_cases_per_report} 条 bad case）"
            )

            try:
                out_dir = run_stage2_dimension_diagnosis(
                    stage1_dir=str(bad_cases_dir),
                    dataset_name=ds,
                    synthesis_llm_config=synthesis_llm_config,
                    total_reports=total_reports,
                    bad_cases_per_report=bad_cases_per_report,
                    output_dir=str(output_path / "_intermediate" / "diagnosis_reports"),
                    batch_size=diagnosis_batch_size,
                )
                report_file = out_dir / "dimension_reports.jsonl"
                with open(report_file, encoding="utf-8") as f:
                    n_reports = sum(1 for _ in f)
                all_stats.setdefault(ds, {})["reports"] = n_reports
                logger.info(f"  ✅ [{ds}] 生成 {n_reports} 份诊断报告")
            except Exception as e:
                logger.error(f"  ❌ [{ds}] 诊断失败: {e}")
                traceback.print_exc()

    # ── Stage 3: 数据合成 ─────────────────────────────────────────────────────
    if stage in ("all", "synth"):
        logger.info(f"\n{'='*70}")
        logger.info("🎨 Stage 3: 根据诊断报告生成新训练数据")
        logger.info(f"{'='*70}")
        from feedback.stage3_synthesis import run_stage3_synthesis

        for ds_info in synthesis_datasets:
            ds = ds_info["name"]
            reports_dir = output_path / "_intermediate" / "diagnosis_reports" / ds

            if not reports_dir.exists():
                logger.warning(f"  ⚠️  [{ds}] 未找到诊断报告目录，跳过合成")
                continue

            # 诊断报告路径查找逻辑：
            # 1. 优先查找 reports_dir/bad_cases/dimension_reports.jsonl
            # 2. 其次查找 reports_dir/dimension_reports.jsonl
            # 3. 如果都不存在，查找 reports_dir 下的第一个子目录
            # 这样处理是为了兼容不同版本的输出结构
            report_path = reports_dir / "bad_cases" / "dimension_reports.jsonl"
            if not report_path.exists():
                report_path = reports_dir / "dimension_reports.jsonl"
                if not report_path.exists():
                    sub_dirs = [p for p in reports_dir.iterdir() if p.is_dir()]
                    if sub_dirs:
                        reports_dir = sub_dirs[0]
                        report_path = reports_dir / "dimension_reports.jsonl"

            if not report_path.exists():
                logger.warning(f"  ⚠️  [{ds}] 未找到诊断报告文件，跳过合成")
                continue

            samples_per_report = ds_info.get("samples_per_report", 5)

            with open(report_path, encoding="utf-8") as f:
                n_actual_reports = sum(1 for _ in f if _.strip())

            logger.info(f"  🔧 [{ds}] 从 {n_actual_reports} 份报告生成数据（每份 {samples_per_report} 条）")

            try:
                out_path = run_stage3_synthesis(
                    reports_dir=str(reports_dir.parent if report_path.parent.name == "bad_cases" else reports_dir),
                    dataset_name=ds,
                    synthesis_llm_config=synthesis_llm_config,
                    samples_per_report=samples_per_report,
                    max_retries=max_retries,
                    output_dir=str(output_path / "_intermediate" / "synth_raw"),
                    model_name=model_name,
                    batch_size=synthesis_batch_size,
                )
                if out_path:
                    with open(out_path, encoding="utf-8") as f:
                        n_synth = sum(1 for _ in f)
                    all_stats.setdefault(ds, {})["synthesized_raw"] = n_synth
                    logger.info(f"  ✅ [{ds}] 生成 {n_synth} 条候选数据")
            except Exception as e:
                logger.error(f"  ❌ [{ds}] 合成失败: {e}")
                traceback.print_exc()

    # ── Stage 4: 守门员过滤（双重去重）───────────────────────────────────────
    if stage in ("all", "synth", "dedupe"):
        logger.info(f"\n{'='*70}")
        logger.info("🛡️  Stage 4: 去重过滤（测试集去重 + 内部去重）")
        logger.info(f"{'='*70}")
        from feedback.stage4_lsh_filter import run_stage4_lsh_filter

        lsh_stats = run_stage4_lsh_filter(config, synthesis_datasets, output_path)
        for ds, st in lsh_stats.items():
            all_stats.setdefault(ds, {}).update(st)

    # ── 生成汇总报告 ──────────────────────────────────────────────────────────
    from feedback.report_summary import save_summary_report
    try:
        save_summary_report(
            stats=all_stats,
            dataset_names=dataset_names,
            stage=stage,
            model_name=model_name,
            output_path=output_path,
        )
    except Exception as e:
        logger.error(f"生成汇总报告失败: {e}")

    logger.info(f"\n{'='*70}")
    logger.info("🎉 流水线完成")
    logger.info(f"{'='*70}")
    for ds, st in all_stats.items():
        logger.info(f"  [{ds}]")
        for key, val in st.items():
            logger.info(f"    • {key}: {val}")
    logger.info(f"{'='*70}\n")



def main():
    parser = argparse.ArgumentParser(description="ToMEval 数据合成流水线")
    parser.add_argument("--config", default="feedback/config.yaml", help="配置文件路径")
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
