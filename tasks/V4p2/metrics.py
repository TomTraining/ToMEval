"""V4p2 分组指标：按维度层级 / 题型 / 视角 / 变体切面统计准确率，
并额外汇总 Q4 的平均 rubric 得分（总体 + 按维度）。"""

from __future__ import annotations

from typing import Any, Dict, List

from src.evaluation.task_metrics import (
    first_value,
    generic_group_metrics,
    meta,
    safe_div,
)


def _q4_score_summary(
    records: List[Dict[str, Any]],
    per_sample_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Q4 rubric 平均分：总体 + 按三级维度。"""
    overall: List[float] = []
    by_dim: Dict[str, List[float]] = {}
    for record, result in zip(records, per_sample_results):
        score = result.get("judge_score")
        if record.get("prompt_type") != "open" or score is None:
            continue
        overall.append(float(score))
        dim = first_value(meta(record).get("dim"))
        by_dim.setdefault(dim, []).append(float(score))
    if not overall:
        return {}
    return {
        "q4_mean_score": sum(overall) / len(overall),
        "q4_count": len(overall),
        "q4_mean_score_by_dim": {k: sum(v) / len(v) for k, v in sorted(by_dim.items())},
    }


def compute_metrics(
    records: List[Dict[str, Any]],
    per_sample_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    metrics = generic_group_metrics(
        records,
        per_sample_results,
        [
            ("by_dim1", lambda r: [first_value(meta(r).get("dim1"))]),
            ("by_dim2", lambda r: [first_value(meta(r).get("dim2"))]),
            ("by_dim3", lambda r: [first_value(meta(r).get("dim"))]),
            ("by_qtype", lambda r: [first_value(meta(r).get("qtype"))]),
            ("by_perspective", lambda r: [first_value(meta(r).get("perspective"))]),
            ("by_variant", lambda r: [first_value(meta(r).get("variant"))]),
            # 长/短文本情景：meta.length_mode 取 long/short。
            ("by_length", lambda r: [first_value(meta(r).get("length_mode"))]),
            # 组合键 "维度|题型"：可视化模块据 "|" 自动透视成热力图。
            ("by_dim3_qtype", lambda r: [f"{first_value(meta(r).get('dim'))}|{first_value(meta(r).get('qtype'))}"]),
        ],
    )
    metrics.update(_q4_score_summary(records, per_sample_results))
    return metrics
