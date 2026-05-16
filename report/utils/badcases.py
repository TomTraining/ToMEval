from __future__ import annotations

import json
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional

from .results import load_metrics_payload, load_prediction_records

_ID_FIELDS = {"id", "Index", "filename", "file", "sample_id", "question_id"}


def _group_key(meta: Dict[str, Any]) -> str:
    if not meta:
        return "unknown"
    parts = []
    for key, value in sorted(meta.items()):
        if key in _ID_FIELDS:
            continue
        if isinstance(value, list):
            parts.append(f"{key}={','.join(str(item) for item in value)}")
        else:
            parts.append(f"{key}={value}")
    return "|".join(parts) if parts else "unknown"


def _group_display(meta: Dict[str, Any]) -> str:
    for key in ("ability", "dimension", "task_type", "question_type"):
        value = meta.get(key)
        if not value:
            continue
        if isinstance(value, list):
            return ", ".join(str(item) for item in value)
        return str(value)
    return _group_key(meta)


def _gold_answer(record: Dict[str, Any]) -> str:
    if record.get("prompt_type") == "open":
        return json.dumps(record.get("correct_answers", []), ensure_ascii=False)
    return json.dumps(record.get("correct_letters", []), ensure_ascii=False)


def load_bad_cases(exp_dir, limit: int, seed: int = 42) -> List[Dict[str, Any]]:
    metrics_payload = load_metrics_payload(exp_dir)
    prediction_records = load_prediction_records(exp_dir)

    per_sample_by_repeat: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for repeat_index, repeat_metrics in enumerate(metrics_payload.get("all_metrics", [])):
        per_sample_by_repeat[repeat_index] = {
            str(item["sample_id"]): item for item in repeat_metrics.get("per_sample_results", [])
        }

    by_sample_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    display_record: Dict[str, Dict[str, Any]] = {}
    for record in prediction_records:
        sample_id = str(record["sample_id"])
        repeat = int(record["repeat"])
        judge_result = per_sample_by_repeat.get(repeat, {}).get(sample_id, {})
        merged = {
            **record,
            "judge_result": judge_result,
            "is_correct": bool(judge_result.get("is_correct")),
        }
        by_sample_id[sample_id].append(merged)
        if repeat == 0:
            display_record[sample_id] = merged

    if not by_sample_id:
        return []

    sample_stats: List[Dict[str, Any]] = []
    group_totals: Dict[str, int] = defaultdict(int)
    group_wrongs: Dict[str, int] = defaultdict(int)

    for sample_id, records in by_sample_id.items():
        rep0 = display_record.get(sample_id) or records[0]
        meta = rep0.get("meta", {}) or {}
        group = _group_key(meta)
        group_totals[group] += 1
        wrong_count = sum(1 for record in records if not record["is_correct"])
        if wrong_count > 0:
            group_wrongs[group] += 1
        sample_stats.append(
            {
                "sample_id": sample_id,
                "display_record": rep0,
                "wrong_count": wrong_count,
                "max_repeat": len(records),
                "group_key": group,
                "group_display": _group_display(meta),
            }
        )

    for item in sample_stats:
        total = group_totals[item["group_key"]]
        wrong_rate = group_wrongs[item["group_key"]] / total if total else 0.0
        item["group_wrong_rate"] = wrong_rate
        if wrong_rate > 0.7 and item["wrong_count"] == item["max_repeat"]:
            item["tier"] = 1
        elif wrong_rate > 0.5 and item["wrong_count"] * 2 >= item["max_repeat"]:
            item["tier"] = 2
        elif item["wrong_count"] > 0:
            item["tier"] = 3
        else:
            item["tier"] = 99

    bad_samples = [item for item in sample_stats if item["wrong_count"] > 0]
    bad_samples.sort(
        key=lambda item: (
            item["tier"],
            -item["group_wrong_rate"],
            -item["wrong_count"],
            item["sample_id"],
        )
    )

    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for item in bad_samples:
        grouped[item["tier"]].append(item)

    rng = random.Random(seed)
    selected: List[Dict[str, Any]] = []
    for tier in (1, 2, 3):
        bucket = grouped.get(tier, [])
        rng.shuffle(bucket)
        bucket.sort(key=lambda item: (-item["group_wrong_rate"], -item["wrong_count"], item["sample_id"]))
        for item in bucket:
            if len(selected) >= limit:
                break
            selected.append(
                {
                    **item["display_record"],
                    "_tier": item["tier"],
                    "_wrong_count": item["wrong_count"],
                    "_max_repeat": item["max_repeat"],
                    "_group_display": item["group_display"],
                    "_group_wrong_rate": item["group_wrong_rate"],
                    "gold_answer": _gold_answer(item["display_record"]),
                }
            )
        if len(selected) >= limit:
            break

    return selected
