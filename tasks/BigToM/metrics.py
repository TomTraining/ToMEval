from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from src.evaluation.task_metrics import (
    add_dimension,
    hierarchical_metrics,
    make_split,
    meta,
    safe_div,
    value_list,
)


def _normalize_belief_type(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _belief_types(record: Dict[str, Any]) -> List[str]:
    types = [_normalize_belief_type(item) for item in value_list(meta(record).get("dimension"))]
    return types or ["unknown"]


def _pair_key(record: Dict[str, Any]) -> str:
    sample_id = str(meta(record).get("id") or record.get("sample_id") or "")
    if "__" in sample_id:
        prefix, _ = sample_id.rsplit("__", 1)
    else:
        prefix = sample_id
    if prefix.endswith("_true_belief"):
        prefix = prefix[: -len("_true_belief")]
    if prefix.endswith("_false_belief"):
        prefix = prefix[: -len("_false_belief")]
    return prefix or "__unknown_pair__"


def compute_metrics(records: List[Dict[str, Any]], per_sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = hierarchical_metrics(
        records,
        per_sample_results,
        [
            ("condition", lambda r: [str(meta(r).get("condition_type") or "unknown")]),
            ("belief_type", _belief_types),
        ],
    )

    # tb_and_fb：同一故事的 true-belief 与 false-belief 两问都对才算该 pair 通过。
    # 该联合分无法从两个边际 accuracy 反推，作为单 split（overall=全部 pair）的二级维度收进 dimensions。
    pair_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"tb_correct": 0, "tb_total": 0, "fb_correct": 0, "fb_total": 0}
    )
    for record, result in zip(records, per_sample_results):
        belief_types = set(_belief_types(record))
        pair_key = _pair_key(record)
        is_correct = result["is_correct"]
        if "true_belief" in belief_types:
            pair_stats[pair_key]["tb_total"] += 1
            pair_stats[pair_key]["tb_correct"] += int(is_correct)
        if "false_belief" in belief_types:
            pair_stats[pair_key]["fb_total"] += 1
            pair_stats[pair_key]["fb_correct"] += int(is_correct)

    tb_fb_total_pairs = 0
    tb_fb_correct_pairs = 0
    for stats in pair_stats.values():
        if stats["tb_total"] and stats["fb_total"]:
            tb_fb_total_pairs += 1
            if stats["tb_correct"] == stats["tb_total"] and stats["fb_correct"] == stats["fb_total"]:
                tb_fb_correct_pairs += 1

    add_dimension(metrics, "tb_and_fb", {
        "overall": make_split(safe_div(tb_fb_correct_pairs, tb_fb_total_pairs), tb_fb_total_pairs),
    })
    return metrics
