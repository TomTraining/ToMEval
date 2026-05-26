"""
阶段五：难度验证

读取 synth_clean/<dataset>/synthetic_iter*_<model>.parquet，对每条样本用弱模型
(默认 qwen3-8b @ DashScope) 跑 N 次推理。任意一次答错 → 保留；全部答对 → 丢弃。
保留的样本写入同目录 synthetic_iter*_<model>_hard.parquet。

字段约定：合成数据使用与正式评测一致的小写字段 (story / question / answer.{correct,wrong}_answers / meta.id)，
直接复用 src/evaluation/prompts.py 的 build_option_bundle + build_prompt + extract_prediction_value，
确保"训练样本难度"与"评测同款 prompt 下的难度"语义对齐。
"""

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import runner
from src.evaluation.prompts import (
    build_option_bundle,
    build_prompt,
    extract_prediction_value,
    prompt_type,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _extract_report_idx(sample_id: str) -> int:
    """从 synthetic_<dataset>_iter<iter>_<report_idx>_<sample_idx> 提取 report_idx。"""
    m = re.search(r"_iter\d+_(\d+)_\d+$", str(sample_id))
    return int(m.group(1)) if m else -1


def _is_correct(
    sample: Dict[str, Any],
    correct_letters: List[str],
    response,
) -> bool:
    pt = prompt_type(sample["answer"])
    pred = extract_prediction_value(pt, response)
    if pt == "mcq_single":
        return pred in correct_letters
    if pt == "mcq_multi":
        return pred is not None and set(pred) == set(correct_letters)
    # open QA：合成数据集几乎不会走这里；做最宽容的文本包含判断
    if pred is None:
        return False
    pred_norm = str(pred).strip().lower()
    return any(
        str(c).strip().lower() in pred_norm
        for c in sample["answer"].get("correct_answers", [])
    )


def filter_by_difficulty(
    parquet_path: Path,
    dataset_name: str,
    llm_config: Dict[str, Any],
    strong_llm_config: Dict[str, Any] = None,
    repeats: int = 3,
    strong_repeats: int = 0,
    keep_when_api_fail: bool = False,
) -> Tuple[Path, Dict[str, Any]]:
    """对单个 parquet 跑难度过滤，返回 (输出文件路径, 统计)。"""
    parquet_path = Path(parquet_path)
    rows = pq.read_table(str(parquet_path)).to_pylist()
    n = len(rows)
    out_path = parquet_path.with_name(parquet_path.stem + "_hard.parquet")

    if n == 0:
        pq.write_table(pa.table({}), str(out_path))
        return out_path, {"total": 0, "kept": 0, "dropped_all_correct": 0, "dropped_api_fail": 0, "kept_rate": 0.0}

    client = runner.create_content_client(llm_config, None)
    strong_client = None
    if strong_llm_config and strong_repeats > 0:
        strong_client = runner.create_content_client(strong_llm_config, None)

    any_wrong = [False] * n
    any_correct = [False] * n
    api_fail_count = [0] * n
    report_idx_of_row = [-1] * n
    report_total_counts: Counter = Counter()
    report_high_value_counts: Counter = Counter()

    #获取每条样本对应的 report_idx，并统计每个 report_idx 的样本总数
    for i, row in enumerate(rows):
        sample_id = (row.get("meta") or {}).get("id") or f"synth_{i}"
        report_idx = _extract_report_idx(sample_id)
        report_idx_of_row[i] = report_idx
        if report_idx >= 0:
            report_total_counts[report_idx] += 1

    for repeat_idx in range(repeats):
        prompts: List[str] = []
        per_sample_ctx: List[Tuple[Dict[str, Any], Dict[str, str], List[str]]] = []

        for i, row in enumerate(rows):
            sample = {
                "story": row.get("story", ""),
                "question": row.get("question", ""),
                "answer": row.get("answer", {"correct_answers": [], "wrong_answers": []}),
            }
            sample_id = (row.get("meta") or {}).get("id") or f"synth_{i}"
            option_map, correct_letters, _, _ = build_option_bundle(
                dataset_name, str(sample_id), sample["answer"], repeat_idx
            )
            prompts.append(build_prompt(sample, option_map))
            per_sample_ctx.append((sample, option_map, correct_letters))

        responses = client.batch_generate(
            prompts, desc=f"Difficulty[{dataset_name}] r{repeat_idx + 1}/{repeats}"
        )

        for i, (resp, (sample, _option_map, correct_letters)) in enumerate(zip(responses, per_sample_ctx)):
            if resp is None or resp.content is None:
                api_fail_count[i] += 1
                continue
            if _is_correct(sample, correct_letters, resp):
                any_correct[i] = True
            else:
                any_wrong[i] = True

    strong_valid_count = [0] * n
    strong_any_correct = [False] * n
    strong_targets = [i for i in range(n) if any_wrong[i] and not any_correct[i]]

    #对于弱模型全部答错的样本，用强模型再测一次，如果强模型多次有效答错
    if strong_client is not None and strong_targets:
        for repeat_idx in range(strong_repeats):
            prompts: List[str] = []
            per_sample_ctx: List[Tuple[int, Dict[str, Any], List[str]]] = []

            for i in strong_targets:
                row = rows[i]
                sample = {
                    "story": row.get("story", ""),
                    "question": row.get("question", ""),
                    "answer": row.get("answer", {"correct_answers": [], "wrong_answers": []}),
                }
                sample_id = (row.get("meta") or {}).get("id") or f"synth_{i}"
                option_map, correct_letters, _, _ = build_option_bundle(
                    dataset_name, str(sample_id), sample["answer"], repeat_idx
                )
                prompts.append(build_prompt(sample, option_map))
                per_sample_ctx.append((i, sample, correct_letters))

            responses = strong_client.batch_generate(
                prompts, desc=f"StrongDifficulty[{dataset_name}] r{repeat_idx + 1}/{strong_repeats}"
            )

            for resp, (i, sample, correct_letters) in zip(responses, per_sample_ctx):
                if resp is None or resp.content is None:
                    continue
                strong_valid_count[i] += 1
                if _is_correct(sample, correct_letters, resp):
                    strong_any_correct[i] = True

    kept_rows: List[Dict[str, Any]] = []
    dropped_correct = 0
    dropped_invalid_question = 0
    dropped_apifail = 0
    #筛选最终保留的样本
    for i, row in enumerate(rows):
        is_high_value = False
        if any_correct[i]:
            if any_wrong[i]:
                kept_rows.append(row)
            else:
                dropped_correct += 1
        elif any_wrong[i]:
            if strong_client is not None and strong_repeats > 0:
                if strong_valid_count[i] > 0 and not strong_any_correct[i]:
                    dropped_invalid_question += 1
                else:
                    kept_rows.append(row)
                #高价值样本：弱模型全部答错且强模型答对
                is_high_value = strong_any_correct[i]
            else:
                kept_rows.append(row)
        elif api_fail_count[i] >= repeats:
            if keep_when_api_fail:
                kept_rows.append(row)
            else:
                dropped_apifail += 1
        else:
            kept_rows.append(row)

        if is_high_value and report_idx_of_row[i] >= 0:
            report_high_value_counts[report_idx_of_row[i]] += 1

    if kept_rows:
        pq.write_table(pa.Table.from_pylist(kept_rows), str(out_path))
    else:
        # 写一个表头一致的空 parquet，避免下游 glob 拿到一个 0 字节文件
        empty_table = pa.Table.from_pylist([dict(rows[0])]).slice(0, 0) if rows else pa.table({})
        pq.write_table(empty_table, str(out_path))

    stats = {
        "total": n,
        "kept": len(kept_rows),
        "dropped_all_correct": dropped_correct,
        "dropped_invalid_question": dropped_invalid_question,
        "dropped_api_fail": dropped_apifail,
        "kept_rate": len(kept_rows) / n,
        "_report_total_counts": dict(report_total_counts),
        "_report_high_value_counts": dict(report_high_value_counts),
    }
    logger.info(
        f"  {dataset_name}/{parquet_path.name}: kept {len(kept_rows)}/{n} "
        f"({stats['kept_rate']:.1%}); dropped_correct={dropped_correct}, "
        f"dropped_invalid={dropped_invalid_question}, api_fail={dropped_apifail}"
    )
    return out_path, stats


def run_stage5_all_datasets(
    config: Dict[str, Any],
    only_dataset: str = "",
    iteration: int = 1,
) -> Dict[str, Dict[str, Any]]:
    """对 synth_clean 下所有匹配 iter 的 parquet 跑难度过滤。"""
    df_cfg = config.get("difficulty_filter", {})
    if not df_cfg.get("enabled", False):
        logger.info("difficulty_filter.enabled=false, skipping Stage 5")
        return {}

    repeats = df_cfg.get("repeats", 3)
    strong_repeats = df_cfg.get("strong_repeats", 1)
    keep_when_api_fail = df_cfg.get("keep_when_api_fail", False)

    output_path = Path(config["output_path"])
    synth_clean_root = output_path / "synth_clean"

    datasets_cfg = config["synthesis_datasets"]

    all_stats: Dict[str, Dict[str, Any]] = {}

    for ds_info in datasets_cfg:
        ds = ds_info["name"]
        if only_dataset and ds != only_dataset:
            continue
        ds_dir = synth_clean_root / ds
        if not ds_dir.exists():
            continue
        report_total_counts: Counter = Counter()
        report_high_value_counts: Counter = Counter()
        for pq_file in sorted(ds_dir.glob(f"synthetic_iter{iteration}_*.parquet")):
            # 跳过自身产物，避免对 _hard.parquet 重复跑
            if pq_file.stem.endswith("_hard"):
                continue
            try:
                _, st = filter_by_difficulty(
                    pq_file, ds, df_cfg,
                    strong_llm_config=config.get("synthesis_model", {}),
                    repeats=repeats,
                    strong_repeats=strong_repeats,
                    keep_when_api_fail=keep_when_api_fail,
                )
                all_stats[f"{ds}/{pq_file.name}"] = st
                report_total_counts.update(st.get("_report_total_counts", {}))
                report_high_value_counts.update(st.get("_report_high_value_counts", {}))
            except Exception as e:
                logger.error(f"difficulty filter failed for {pq_file}: {e}")
                import traceback
                traceback.print_exc()

        #读取诊断报告，找出高价值样本对应的 report_idx
        reports_root = Path(config["output_path"]) / "diagnosis_reports" / ds
        reports_path = None
        if reports_root.exists():
            candidate = reports_root / "dimension_reports.jsonl"
            if candidate.exists():
                reports_path = candidate
            else:
                for sub_dir in [p for p in reports_root.iterdir() if p.is_dir()]:
                    candidate = sub_dir / "dimension_reports.jsonl"
                    if candidate.exists():
                        reports_path = candidate
                        break

        if reports_path and report_total_counts:
            with open(reports_path, encoding="utf-8") as f:
                base_reports = [json.loads(line) for line in f if line.strip()]

            feedback_rows = []
            for idx, report in enumerate(base_reports):
                total = int(report_total_counts.get(idx, 0))
                if total <= 0:
                    continue
                high_value = int(report_high_value_counts.get(idx, 0))
                rate = high_value / total
                if rate >= 0.3:
                    row = dict(report)
                    row["high_value_count"] = high_value
                    row["high_value_rate"] = rate
                    feedback_rows.append(row)

            # 将高价值样本对应的诊断报告保存，供后续迭代
            if feedback_rows:
                logger.info(f"  {ds}: {len(feedback_rows)}/{len(base_reports)} reports have high-value samples (rate>=0.5), saving feedback")
                print("high-value report")
                feedback_dir = reports_root / f"stage5_feedback_iter{iteration}"
                feedback_dir.mkdir(parents=True, exist_ok=True)
                feedback_path = feedback_dir / "dimension_reports.jsonl"
                with open(feedback_path, "w", encoding="utf-8") as f:
                    for row in feedback_rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                logger.info(f"  {ds}: saved {len(feedback_rows)} feedback reports to {feedback_path}")

    return all_stats
