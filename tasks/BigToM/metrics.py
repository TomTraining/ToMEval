from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from src.evaluation.task_metrics import (
    base_metric_payload,
    count_dict,
    flatten_group,
    meta,
    rate_dict,
    safe_div,
    update_group,
    value_list,
)


def _normalize_belief_type(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


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
    metrics = base_metric_payload(per_sample_results)
    by_condition: Dict[str, Dict[str, int]] = {}
    by_belief_type: Dict[str, Dict[str, int]] = {}
    pair_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"tb_correct": 0, "tb_total": 0, "fb_correct": 0, "fb_total": 0}
    )

    for record, result in zip(records, per_sample_results):
        record_meta = meta(record)
        is_correct = result["is_correct"]

        update_group(by_condition, record_meta.get("condition_type"), is_correct)

        belief_types = {_normalize_belief_type(item) for item in value_list(record_meta.get("dimension"))}
        if not belief_types:
            belief_types = {"unknown"}
        for belief_type in belief_types:
            update_group(by_belief_type, belief_type, is_correct)

        pair_key = _pair_key(record)
        if "true_belief" in belief_types:
            pair_stats[pair_key]["tb_total"] += 1
            if is_correct:
                pair_stats[pair_key]["tb_correct"] += 1
        if "false_belief" in belief_types:
            pair_stats[pair_key]["fb_total"] += 1
            if is_correct:
                pair_stats[pair_key]["fb_correct"] += 1

    tb_fb_total_pairs = 0
    tb_fb_correct_pairs = 0
    for stats in pair_stats.values():
        if stats["tb_total"] and stats["fb_total"]:
            tb_fb_total_pairs += 1
            if stats["tb_correct"] == stats["tb_total"] and stats["fb_correct"] == stats["fb_total"]:
                tb_fb_correct_pairs += 1

    by_belief_type_rates = rate_dict(by_belief_type)
    by_belief_type_rates["tb_and_fb"] = safe_div(tb_fb_correct_pairs, tb_fb_total_pairs)

    metrics.update(flatten_group(by_condition, "by_condition"))
    metrics.update({f"by_belief_type.{key}": value for key, value in by_belief_type_rates.items()})
    metrics["by_condition"] = rate_dict(by_condition)
    metrics["condition_counts"] = count_dict(by_condition)
    metrics["by_belief_type"] = by_belief_type_rates
    metrics["belief_type_counts"] = {**count_dict(by_belief_type), "tb_and_fb": tb_fb_total_pairs}
    return metrics
