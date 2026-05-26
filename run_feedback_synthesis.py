"""
数据合成流水线入口

流程：
  1. 从 tomeval_predictions_latest_full 加载 bad case（三模型并集）
  2. 维度批量诊断（stage2）
  3. 从诊断报告合成新样本（stage3）
  4. LSH 守门员过滤（merge_and_dedupe）
  5. 写迭代日志（ITERATION_LOG.md）

用法：
  cd /Users/yangmeili/Downloads/Code/ToMEval
  python run_feedback_synthesis.py --stage all --dataset ToMBench --max-bad-cases 80
  python run_feedback_synthesis.py --stage diagnose --dataset BigToM
  python run_feedback_synthesis.py --stage synth
  python run_feedback_synthesis.py --stage all --iteration 2
"""

import argparse
import json
import logging
import math
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

# 确保 src/ 可见
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


def _count_bad_cases_from_files(
    dataset_names: list,
    output_path: Path,
) -> Dict[str, int]:
    """从已存在的 bad_cases.jsonl 读取各数据集 bad case 数量。
    用于跳过 Stage 1 时（--stage diagnose / synth）也能计算跨数据集比例。
    """
    counts: Dict[str, int] = {}
    for ds in dataset_names:
        bc_file = output_path / "bad_cases" / ds / "bad_cases.jsonl"
        if bc_file.exists():
            with open(bc_file, encoding="utf-8") as f:
                counts[ds] = sum(1 for line in f if line.strip())
        else:
            counts[ds] = 0
            logger.warning(f"  bad_cases.jsonl not found for {ds}, treating count as 0")
    return counts


def _compute_proportional_allocations(
    bad_case_counts: Dict[str, int],
    total_reports_budget: int,
    total_samples_budget: int,
) -> Dict[str, Tuple[int, int]]:
    """按各数据集 bad case 数量占比，分配报告配额和合成样本配额。

    Returns:
        {dataset_name: (report_quota, samples_quota)}
    """
    datasets = [ds for ds, n in bad_case_counts.items() if n > 0]
    total = sum(bad_case_counts.get(ds, 0) for ds in datasets)

    if total == 0:
        # 全部为 0 时均分
        n = max(len(datasets), 1)
        return {ds: (max(1, total_reports_budget // n), max(1, total_samples_budget // n))
                for ds in datasets}

    # 按比例向下取整（每个数据集至少 1 份）
    report_raw = {ds: max(1, int(total_reports_budget * bad_case_counts[ds] / total)) for ds in datasets}
    samples_raw = {ds: max(1, int(total_samples_budget * bad_case_counts[ds] / total)) for ds in datasets}

    # 修正舍入差，将剩余配额按 bad case 从多到少补给
    def _fix_rounding(raw: Dict[str, int], budget: int) -> Dict[str, int]:
        diff = budget - sum(raw.values())
        for ds in sorted(raw, key=lambda d: -bad_case_counts.get(d, 0)):
            if diff <= 0:
                break
            raw[ds] += 1
            diff -= 1
        return raw

    report_alloc = _fix_rounding(report_raw, total_reports_budget)
    samples_alloc = _fix_rounding(samples_raw, total_samples_budget)

    # 没有 bad case 的数据集分配 0
    result: Dict[str, Tuple[int, int]] = {}
    for ds in bad_case_counts:
        result[ds] = (report_alloc.get(ds, 0), samples_alloc.get(ds, 0))
    return result


def _count_reports(dataset_name: str, output_path: Path) -> int:
    """读取已生成的 dimension_reports.jsonl 行数，用于 Stage 3 前确定报告数量。"""
    # 新格式路径：diagnosis_reports/<ds>/bad_cases/dimension_reports.jsonl
    for subdir in ["bad_cases", dataset_name, ""]:
        candidate = output_path / "diagnosis_reports" / dataset_name / subdir / "dimension_reports.jsonl"
        if candidate.exists():
            with open(candidate, encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
    return 0


def run_pipeline(args):
    config_path = args.config
    config = load_config(config_path)

    only_dataset = args.dataset or ""
    max_bad_cases = args.max_bad_cases
    iteration = args.iteration
    stage = args.stage
    report_source = getattr(args, "report_source", "diagnosis")

    if report_source == "stage5" and stage != "synth":
        logger.error("--report-source stage5 is only allowed with --stage synth")
        sys.exit(1)

    output_path = Path(config["output_path"])
    synthesis_llm_config = config["synthesis_model"]
    model_name = synthesis_llm_config.get("model_name", "unknown")

    # 只在需要调用合成模型的阶段才检查 API key
    if stage in ("all", "diagnose", "synth") and not synthesis_llm_config.get("api_key"):
        logger.error(
            "synthesis_model.api_key is empty in config.yaml. "
            "Please fill in your DashScope API key."
        )
        sys.exit(1)

    # ── 数据集列表 ────────────────────────────────────────────────────────────
    synthesis_datasets = config["synthesis_datasets"]
    if only_dataset:
        synthesis_datasets = [d for d in synthesis_datasets if d["name"] == only_dataset]
    if not synthesis_datasets:
        logger.error(f"No datasets to process (only_dataset={only_dataset!r})")
        sys.exit(1)

    dataset_names = [d["name"] for d in synthesis_datasets]
    logger.info(f"Stage={stage}, datasets={dataset_names}, iteration={iteration}, max_bad_cases={max_bad_cases}")

    syn_config = config.get("synthesis", {})
    samples_per_batch = syn_config.get("samples_per_batch", 5)
    max_retries = syn_config.get("max_retries_per_diagnosis", 3)
    total_reports_budget = syn_config.get("total_reports_budget", 60)
    total_samples_budget = syn_config.get("total_samples_budget", 300)

    predictions_root = config.get("predictions_root", "results")

    # 汇总统计（用于迭代日志）
    all_stats = {}

    # ── Stage 1: 加载 bad case ───────────────────────────────────────────────
    if stage in ("all", "load"):
        logger.info("\n===== Stage 1: Load Bad Cases =====")
        from feedback_synthesis.stage1_load_predictions import load_bad_cases_from_predictions

        for ds_info in synthesis_datasets:
            ds = ds_info["name"]
            models = ds_info.get("models", [])
            if not models:
                logger.warning(f"  {ds}: no models configured, skipping")
                continue
            try:
                out_dir = load_bad_cases_from_predictions(
                    dataset_name=ds,
                    predictions_root=predictions_root,
                    models=models,
                    max_bad_cases=max_bad_cases,
                    output_dir=str(output_path / "bad_cases"),
                )
                bc_file = out_dir / "bad_cases.jsonl"
                with open(bc_file, encoding="utf-8") as f:
                    n = sum(1 for _ in f)
                all_stats.setdefault(ds, {})["bad_cases"] = n
                logger.info(f"  {ds}: {n} bad cases loaded")
            except Exception as e:
                logger.error(f"  {ds}: load bad cases failed: {e}")
                traceback.print_exc()

    # ── 跨数据集比例分配（Stage 2 / Stage 3 共用）──────────────────────────────
    if stage in ("all", "diagnose", "synth"):
        # 读取各数据集的 bad case 数量（Stage 1 刚写好，或从已有文件读）
        bad_case_counts = _count_bad_cases_from_files(dataset_names, output_path)
        # 从 all_stats 覆盖刚跑完的 Stage 1 结果（更准确）
        for ds, st in all_stats.items():
            if "bad_cases" in st:
                bad_case_counts[ds] = st["bad_cases"]

        allocations = _compute_proportional_allocations(
            bad_case_counts, total_reports_budget, total_samples_budget
        )
        logger.info(
            f"\n===== Budget Allocation =====\n"
            f"  total_reports_budget={total_reports_budget}, "
            f"total_samples_budget={total_samples_budget}"
        )
        for ds in dataset_names:
            rq, sq = allocations.get(ds, (0, 0))
            logger.info(
                f"  {ds}: bad_cases={bad_case_counts.get(ds, 0)}, "
                f"report_quota={rq}, samples_quota={sq}"
            )
    else:
        allocations = {}

    # ── Stage 2: 维度诊断 ─────────────────────────────────────────────────────
    if stage in ("all", "diagnose"):
        logger.info("\n===== Stage 2: Dimension Diagnosis =====")
        from feedback_synthesis.stage2_diagnosis import run_stage2_dimension_diagnosis

        for ds_info in synthesis_datasets:
            ds = ds_info["name"]
            bad_cases_dir = output_path / "bad_cases" / ds
            if not bad_cases_dir.exists():
                logger.warning(f"  {ds}: bad_cases dir not found, skipping diagnosis")
                continue
            report_quota, _ = allocations.get(ds, (None, None))
            try:
                out_dir = run_stage2_dimension_diagnosis(
                    stage1_dir=str(bad_cases_dir),
                    dataset_name=ds,
                    synthesis_llm_config=synthesis_llm_config,
                    samples_per_batch=samples_per_batch,
                    output_dir=str(output_path / "diagnosis_reports"),
                    max_reports=report_quota,
                )
                report_file = out_dir / "dimension_reports.jsonl"
                with open(report_file, encoding="utf-8") as f:
                    n_reports = sum(1 for _ in f)
                all_stats.setdefault(ds, {})["reports"] = n_reports
                logger.info(f"  {ds}: {n_reports} dimension reports generated (quota={report_quota})")
            except Exception as e:
                logger.error(f"  {ds}: diagnosis failed: {e}")
                traceback.print_exc()

    # ── Stage 3: 数据合成 ─────────────────────────────────────────────────────
    if stage in ("all", "synth"):
        logger.info("\n===== Stage 3: Data Synthesis =====")
        from feedback_synthesis.stage3_synthesis import run_stage3_synthesis

        for ds_info in synthesis_datasets:
            ds = ds_info["name"]
            reports_dir = output_path / "diagnosis_reports" / ds

            if not reports_dir.exists():
                logger.warning(f"  {ds}: diagnosis_reports dir not found, skipping synthesis")
                continue

            report_path = reports_dir / "dimension_reports.jsonl"
            if report_source == "stage5":
                if iteration <= 1:
                    logger.warning(f"  {ds}: stage5 feedback requires iteration > 1, skipping synthesis")
                    continue
                feedback_dir = reports_dir / f"stage5_feedback_iter{iteration - 1}"
                feedback_path = feedback_dir / "dimension_reports.jsonl"
                if not feedback_path.exists():
                    logger.warning(f"  {ds}: stage5 feedback not found, skipping synthesis")
                    continue
                reports_dir = feedback_dir
                report_path = feedback_path
            elif not report_path.exists():
                sub_dirs = [p for p in reports_dir.iterdir() if p.is_dir()]
                if sub_dirs:
                    reports_dir = sub_dirs[0]
                    report_path = reports_dir / "dimension_reports.jsonl"

            if not report_path.exists():
                logger.warning(f"  {ds}: dimension_reports.jsonl not found, skipping synthesis")
                continue

            # 计算该数据集的 samples_per_report（按配额均匀分配到每份报告）
            _, samples_quota = allocations.get(ds, (None, None))
            with open(report_path, encoding="utf-8") as f:
                n_actual_reports = sum(1 for _ in f if _.strip())
            if samples_quota and n_actual_reports > 0:
                ds_samples_per_report = max(1, math.ceil(samples_quota / n_actual_reports))
            else:
                # fallback：无预算配置时默认每份报告生成 2 条
                ds_samples_per_report = 2
            logger.info(
                f"  {ds}: reports={n_actual_reports}, samples_quota={samples_quota}, "
                f"samples_per_report={ds_samples_per_report}"
            )

            try:
                out_path = run_stage3_synthesis(
                    reports_dir=str(reports_dir),
                    dataset_name=ds,
                    synthesis_llm_config=synthesis_llm_config,
                    samples_per_report=ds_samples_per_report,
                    max_retries=max_retries,
                    output_dir=str(output_path / "synth_raw"),
                    model_name=model_name,
                    iteration=iteration,
                )
                if out_path:
                    with open(out_path, encoding="utf-8") as f:
                        n_synth = sum(1 for _ in f)
                    all_stats.setdefault(ds, {})["synthesized_raw"] = n_synth
                    logger.info(f"  {ds}: {n_synth} raw candidates generated → {out_path}")
            except Exception as e:
                logger.error(f"  {ds}: synthesis failed: {e}")
                traceback.print_exc()

    # ── Stage 4: 守门员过滤 ───────────────────────────────────────────────────
    if stage in ("all", "synth", "dedupe"):
        logger.info("\n===== Stage 4: LSH Leakage Guard =====")
        from feedback_synthesis.stage4_lsh_filter import run_stage4_lsh_filter

        lsh_stats = run_stage4_lsh_filter(config, synthesis_datasets, output_path, iteration)
        for ds, st in lsh_stats.items():
            all_stats.setdefault(ds, {}).update(st)

    # ── Stage 5: 难度验证（qwen3-8b 至少错 1 次才保留）─────────────────────────
    if stage in ("all", "synth", "dedupe", "difficulty"):
        logger.info("\n===== Stage 5: Difficulty Filter (qwen3-8b) =====")
        from feedback_synthesis.stage5_difficulty_filter import run_stage5_all_datasets

        diff_stats = run_stage5_all_datasets(
            config, only_dataset=only_dataset, iteration=iteration
        )
        for key, st in diff_stats.items():
            ds = key.split("/")[0]
            all_stats.setdefault(ds, {})["difficulty_kept"] = st["kept"]
            all_stats.setdefault(ds, {})["difficulty_dropped_correct"] = st["dropped_all_correct"]
            all_stats.setdefault(ds, {})["difficulty_dropped_invalid"] = st["dropped_invalid_question"]
            all_stats.setdefault(ds, {})["difficulty_api_fail"] = st["dropped_api_fail"]

    # ── 写迭代日志 ────────────────────────────────────────────────────────────
    _write_iteration_log(
        config=config,
        iteration=iteration,
        stage=stage,
        dataset_names=dataset_names,
        stats=all_stats,
        output_path=output_path,
    )

    logger.info("\n===== Pipeline Complete =====")
    for ds, st in all_stats.items():
        logger.info(f"  {ds}: {st}")


def _write_iteration_log(config, iteration, stage, dataset_names, stats, output_path):
    """追加写入 ITERATION_LOG.md"""
    log_path = Path(__file__).parent / "feedback_synthesis" / "ITERATION_LOG.md"
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    syn_cfg = config.get("synthesis", {})
    model_name = config.get("synthesis_model", {}).get("model_name", "unknown")

    section_lines = [
        f"\n## Iteration {iteration} — {now}",
        f"**Stage**: {stage}",
        f"**Datasets**: {', '.join(dataset_names)}",
        f"**Synthesis model**: {model_name}",
        f"**samples_per_batch**: {syn_cfg.get('samples_per_batch', 5)} | "
        f"**samples_per_report**: {syn_cfg.get('samples_per_report', 2)} | "
        f"**max_retries**: {syn_cfg.get('max_retries_per_diagnosis', 3)}",
        "",
        "### Stats",
    ]
    for ds, st in stats.items():
        section_lines.append(
            f"- **{ds}**: bad_cases={st.get('bad_cases','?')} | "
            f"reports={st.get('reports','?')} | "
            f"raw={st.get('synthesized_raw','?')} | "
            f"clean={st.get('synth_clean','?')} | "
            f"dropped={st.get('synth_dropped','?')} | "
            f"hard={st.get('difficulty_kept','?')} | "
            f"dropped_correct={st.get('difficulty_dropped_correct','?')} | "
            f"dropped_invalid={st.get('difficulty_dropped_invalid','?')}"
        )

    # 占位符（首次运行后人工填写）
    synth_clean_root = output_path / "synth_clean"
    samples_text = "\n### Samples\n*(run complete then inspect data_output/synth_clean/ and paste 3 examples here)*\n"
    gaps_text = "\n### Gaps\n*(to be filled after manual review)*\n"
    next_text = "\n### Next\n*(to be filled after evaluation)*\n"

    # 首次运行时自动放 3 条样例
    sample_rows = []
    for ds in dataset_names:
        ds_clean_dir = synth_clean_root / ds
        if not ds_clean_dir.exists():
            continue
        for pq_file in sorted(ds_clean_dir.glob("synthetic_iter*.parquet")):
            try:
                import pyarrow.parquet as pq_mod
                rows = pq_mod.read_table(str(pq_file)).to_pylist()
                for r in rows[:3]:
                    sample_rows.append(f"**{ds}** — {json.dumps(r, ensure_ascii=False)[:300]}")
                if sample_rows:
                    break
            except Exception:
                pass
        if sample_rows:
            break

    if sample_rows:
        samples_text = "\n### Samples (auto)\n" + "\n".join(f"- {s}" for s in sample_rows) + "\n"

    content = "\n".join(section_lines) + samples_text + gaps_text + next_text + "\n---\n"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"Iteration log updated: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="ToMEval 数据合成流水线")
    parser.add_argument(
        "--stage",
        choices=["all", "load", "diagnose", "synth", "dedupe", "difficulty"],
        default="dedupe",
        help="运行阶段: all=全流程, load=只加载bad case, diagnose=诊断, synth=合成+守门员+难度, difficulty=只跑难度过滤",
    )
    parser.add_argument("--config", default="feedback_synthesis/config.yaml", help="配置文件路径")
    parser.add_argument("--dataset", default="HiToM", help="只运行单个数据集，如 ToMBench")
    parser.add_argument("--max-bad-cases", type=int, default=80, help="每数据集最多 bad case 数（0=不限）")
    parser.add_argument("--iteration", type=int, default=6, help="迭代轮次，影响输出文件命名")
    parser.add_argument(
        "--report-source",
        choices=["diagnosis", "stage5"],
        default="diagnosis",
        help="Stage 3 读取哪类报告：diagnosis=原始 dimension_reports.jsonl，stage5=上一轮 Stage 5 反馈报告",
    )
    args = parser.parse_args()

    run_pipeline(args)


if __name__ == "__main__":
    main()
