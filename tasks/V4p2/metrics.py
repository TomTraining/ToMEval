"""V4p2 分组指标：按维度层级 / 题型 / 视角 / 变体切面统计准确率，
并额外汇总 Q4 的平均 rubric 得分（总体 + 按维度）。

二级指标 `qualified`：只在「人工审核合格」样本（meta.review_pass=True）上重算同一套指标，
用于对比剔除审核不合格题前后的评测结果。审核标记由 scripts/v4p2_apply_review_flags.py
预先写入数据集 meta；缺失该字段的样本默认视为合格。"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.evaluation.task_metrics import (
    first_value,
    generic_group_metrics,
    meta,
    safe_div,
)

# 分组切面定义：full 与 qualified 二级指标共用同一套切面。
GROUP_SPECS: List[Tuple[str, Any]] = [
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
]


def _is_qualified(record: Dict[str, Any]) -> bool:
    """meta.review_pass 缺失时默认合格，保证未打标数据/其他数据集不受影响。"""
    return bool(meta(record).get("review_pass", True))


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


def _block(
    records: List[Dict[str, Any]],
    per_sample_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """一套完整指标（分组准确率 + Q4 平均分）。"""
    block = generic_group_metrics(records, per_sample_results, GROUP_SPECS)
    block.update(_q4_score_summary(records, per_sample_results))
    return block


def compute_metrics(
    records: List[Dict[str, Any]],
    per_sample_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    metrics = _block(records, per_sample_results)

    # 二级指标：仅人工审核合格样本（review_pass=True）。
    pairs = [
        (r, res) for r, res in zip(records, per_sample_results) if _is_qualified(r)
    ]
    qualified_records = [p[0] for p in pairs]
    qualified_results = [p[1] for p in pairs]

    total = len(records)
    n_qualified = len(qualified_records)
    qualified = _block(qualified_records, qualified_results)
    # 二级指标自带的 per_sample_results 与顶层重复，去掉以免膨胀。
    qualified.pop("per_sample_results", None)

    metrics["review_pass_count"] = n_qualified
    metrics["review_fail_count"] = total - n_qualified
    metrics["review_pass_rate"] = safe_div(n_qualified, total)
    metrics["qualified"] = qualified
    return metrics
