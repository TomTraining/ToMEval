"""del_tom 协议的多数投票聚合。

del_tom 复用 repeats 机制对同一 sample 跑 n_samples=8 次(且不 shuffle，选项顺序一致)，
本模块把同一 sample 的多次结果折叠成「每样本一条」的判分结果，供 metric 阶段聚合。

- mcq_single(主路径):每次 extract_cot 得字母 → 多数投票(平局取字母序最小)。
- mcq_multi(退化):逐字母严格多数(某字母在 > n/2 个 repeat 的 boxed 集合中才计入)。
- open(退化):文本投票不可靠，取 repeat 最小的代表样本走正常 open LLM judge。
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from typing import Any, Dict, List, Optional

from .judge import backfill_meta, judge_repeat
from .prompts import extract_prediction_from_text
from .storage import pred_content


def majority_vote_letter(letters: List[Optional[str]]) -> Optional[str]:
    """出现最多的非 None 字母；平局取字母序最小；全为 None 时返回 None。"""
    valid = [letter for letter in letters if letter]
    if not valid:
        return None
    counts = Counter(valid)
    top = max(counts.values())
    return sorted(letter for letter, count in counts.items() if count == top)[0]


def _vote_single(group: List[Dict[str, Any]], rep: Dict[str, Any]) -> Dict[str, Any]:
    letters: List[Optional[str]] = []
    for record in group:
        content = pred_content(record)
        if content in (None, ""):
            letters.append(None)
            continue
        extractor = record.get("extractor", "cot")
        letters.append(extract_prediction_from_text("mcq_single", str(content), extractor))

    voted = majority_vote_letter(letters)
    if voted is None:
        return {"is_correct": False, "error_reason": "extraction_failed", "extracted": None, "votes": letters}
    correct_letters = rep.get("correct_letters") or []
    is_correct = voted in correct_letters
    return {
        "is_correct": is_correct,
        "error_reason": None if is_correct else "wrong_answer",
        "extracted": voted,
        "votes": letters,
    }


def _vote_multi(group: List[Dict[str, Any]], rep: Dict[str, Any]) -> Dict[str, Any]:
    n = len(group)
    counts: Counter = Counter()
    valid = 0
    for record in group:
        content = pred_content(record)
        if content in (None, ""):
            continue
        extractor = record.get("extractor", "cot")
        extracted = extract_prediction_from_text("mcq_multi", str(content), extractor)
        if not extracted:
            continue
        valid += 1
        for letter in set(extracted):
            counts[letter] += 1

    if valid == 0:
        return {"is_correct": False, "error_reason": "extraction_failed", "extracted": None}
    # 严格多数:某字母出现在超过半数(> n/2)的 repeat 里才计入最终答案集合。
    voted_set = sorted(letter for letter, count in counts.items() if count > n / 2)
    correct_letters = rep.get("correct_letters") or []
    is_correct = set(voted_set) == set(correct_letters)
    return {
        "is_correct": is_correct,
        "error_reason": None if is_correct else "wrong_answer",
        "extracted": voted_set,
    }


def vote_collapse(records: List[Dict[str, Any]], open_ctx: Any = None):
    """把同一 sample 的多次 repeat 折叠成每样本一条 (代表 record, 判分结果)。

    open_ctx:OpenJudgeContext，open 题的退化判分按其模式分派(f1/llm_simple/rubric)。
    返回 (voted_records, voted_results),两者等长且一一对应,可直接喂给 aggregate_metrics。
    """
    groups: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for record in records:
        groups.setdefault(record["sample_id"], []).append(record)

    voted_records: List[Dict[str, Any]] = []
    voted_results: List[Optional[Dict[str, Any]]] = []
    open_reps: List[Dict[str, Any]] = []
    open_positions: List[int] = []

    for sample_id, group in groups.items():
        # 代表样本取 repeat 最小的一条，保证 meta / 分组键稳定可复现。
        rep = min(group, key=lambda record: int(record["repeat"]))
        ptype = rep["prompt_type"]
        voted_records.append(rep)
        if ptype == "mcq_single":
            voted_results.append(_vote_single(group, rep))
        elif ptype == "mcq_multi":
            voted_results.append(_vote_multi(group, rep))
        else:  # open:退化为对代表样本做一次 LLM judge
            voted_results.append(None)
            open_positions.append(len(voted_results) - 1)
            open_reps.append(rep)

    if open_reps:
        open_results = judge_repeat(open_reps, open_ctx=open_ctx)
        for position, result in zip(open_positions, open_results):
            voted_results[position] = result

    for record, result in zip(voted_records, voted_results):
        backfill_meta(result, record)
    return voted_records, voted_results


__all__ = ["majority_vote_letter", "vote_collapse"]
