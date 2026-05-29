"""F035：双指标难度评估（不融合）。

- 简模型 (qwen3-8b)：repeat=5 实际答题，得 simple_pass_rate ∈ {0/5..5/5}
- 强模型 (deepseek-v4-flash)：DIFFICULTY_PROMPTS 给 0-5 分（strong_difficulty）

records schema 是 F036 硬契约（D035-02），禁止重命名：
  records[i] = {row_idx, meta_id, simple_correct: [bool]*5,
                simple_pass_rate: 0..5, strong_difficulty: 0..5|null}

报告写到 data_eval_output/difficulty/<DS>_<stem>.json。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from data_eval.answerability_core import (
    classify_prompt_type as _classify_prompt_type,
    get_correct_letters as _get_correct_letters,
    build_answer_prompt as _build_answer_prompt,
    check_correct_single as _check_correct_single,
    check_correct_multi as _check_correct_multi,
)
from data_eval.base import EvalResult, load_answer_models, load_sample_rows, load_synth_parquet, write_report
from data_eval.prompts import DIFFICULTY_PROMPTS
from src.llm.content_client import ContentClient
from src.llm.llm_utils import extract_json

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "config.yaml"

# F035 D035-03: 采样固定 random_state；F041：行数从 config.yaml sample_rows 读
_SAMPLE_SEED = 20260527
_SIMPLE_REPEAT = 5


def _load_strong_client() -> ContentClient:
    """F039：强模型 = config.yaml eval_model.strong（deepseek-v4-flash）。"""
    return load_answer_models()["strong"]


def _load_simple_client() -> ContentClient:
    """F039：简模型 = config.yaml eval_model.simple（qwen3-8b）。"""
    return load_answer_models()["simple"]


def _build_strong_prompt(dataset: str, row: Any) -> str:
    template = DIFFICULTY_PROMPTS[dataset]
    answer = row["answer"] if isinstance(row["answer"], dict) else {}
    correct = answer.get("correct_answers", [])
    if isinstance(correct, (list, tuple)) and correct:
        correct_str = ", ".join(str(x) for x in correct)
    else:
        correct_str = str(correct)
    return template.format(
        story=str(row.get("story", "")),
        question=str(row.get("question", "")),
        correct_answers=correct_str,
    )


def _parse_strong(text: str | None) -> int | None:
    if not text:
        return None
    parsed = extract_json(text)
    if not parsed:
        return None
    score = parsed.get("score")
    if isinstance(score, (int, float)):
        s = int(score)
        if 0 <= s <= 5:
            return s
    return None


def run_difficulty_eval_on_df(
    df: Any,
    dataset: str,
    file_stem: str,
    max_rows: int | None = None,
    output_root: str = "data_eval_output",
    output_subdir: str = "difficulty",
) -> EvalResult:
    """F038：对任意 df 跑 F035 双指标难度评估。供 run_train_eval.py 复用。"""
    if dataset not in DIFFICULTY_PROMPTS:
        raise ValueError(f"dataset={dataset!r} 不在 DIFFICULTY_PROMPTS 中")

    sample_size = max_rows if max_rows is not None else load_sample_rows(dataset)
    sample_size = min(sample_size, len(df))
    if sample_size < len(df):
        df = df.sample(n=sample_size, random_state=_SAMPLE_SEED).sort_index()
    else:
        df = df.copy()

    strong_client = _load_strong_client()
    simple_client = _load_simple_client()

    row_indices: list[int] = []
    meta_ids: list[str] = []
    prompt_types: list[str] = []
    correct_letters: list[list[str]] = []
    simple_prompts: list[str] = []
    strong_prompts: list[str] = []

    for idx, row in df.iterrows():
        row_indices.append(int(idx))
        meta = row.get("meta")
        meta_ids.append(str(meta.get("id", "")) if isinstance(meta, dict) else "")
        pt = _classify_prompt_type(row)
        cl = _get_correct_letters(row)
        prompt_types.append(pt)
        correct_letters.append(cl)
        simple_prompts.append(_build_answer_prompt(pt, row))
        strong_prompts.append(_build_strong_prompt(dataset, row))

    n = len(row_indices)

    flat_simple_prompts: list[str] = []
    for k in range(_SIMPLE_REPEAT):
        flat_simple_prompts.extend(simple_prompts)
    simple_responses = simple_client.batch_generate(
        flat_simple_prompts, desc=f"difficulty/simple/{dataset}/{file_stem}"
    )

    simple_correct: list[list[bool]] = [[False] * _SIMPLE_REPEAT for _ in range(n)]
    for k in range(_SIMPLE_REPEAT):
        for i in range(n):
            resp = simple_responses[k * n + i]
            raw = resp.content if resp and resp.content else None
            if raw is None:
                continue
            pt = prompt_types[i]
            cl = correct_letters[i]
            if pt == "mcq_multi":
                simple_correct[i][k] = _check_correct_multi(raw, cl)
            else:
                simple_correct[i][k] = _check_correct_single(raw, cl)

    strong_responses = strong_client.batch_generate(
        strong_prompts, desc=f"difficulty/strong/{dataset}/{file_stem}"
    )
    strong_scores: list[int | None] = [_parse_strong(r.content) for r in strong_responses]

    simple_pass_rate_distribution: dict[str, int] = {f"{k}/5": 0 for k in range(6)}
    strong_difficulty_score_distribution: dict[str, int] = {str(s): 0 for s in range(6)}
    strong_failed_count = 0
    strong_score_sum = 0
    strong_scored_count = 0

    records: list[dict[str, Any]] = []
    for i in range(n):
        s_pass = sum(1 for c in simple_correct[i] if c)
        simple_pass_rate_distribution[f"{s_pass}/5"] += 1
        st = strong_scores[i]
        if st is None:
            strong_failed_count += 1
        else:
            strong_difficulty_score_distribution[str(st)] += 1
            strong_score_sum += st
            strong_scored_count += 1
        records.append({
            "row_idx": row_indices[i],
            "meta_id": meta_ids[i],
            "simple_correct": list(simple_correct[i]),
            "simple_pass_rate": s_pass,
            "strong_difficulty": st,
        })

    simple_mean_pass_rate = round(
        sum(r["simple_pass_rate"] for r in records) / (n * _SIMPLE_REPEAT), 4
    ) if n > 0 else None
    strong_mean = round(strong_score_sum / strong_scored_count, 3) if strong_scored_count > 0 else None

    report: dict[str, Any] = {
        "dataset": dataset,
        "eval_type": "difficulty",
        "total_rows": n,
        "simple_repeat": _SIMPLE_REPEAT,
        "simple_pass_rate_distribution": simple_pass_rate_distribution,
        "simple_mean_pass_rate": simple_mean_pass_rate,
        "strong_difficulty_score_distribution": strong_difficulty_score_distribution,
        "strong_difficulty_mean": strong_mean,
        "strong_failed_count": strong_failed_count,
        "records": records,
    }

    out_path = Path(output_root) / output_subdir / f"{dataset}_{file_stem}.json"
    write_report(report, out_path)

    return EvalResult(
        dataset=dataset,
        eval_type="difficulty",
        total_rows=n,
        pass_=True,
        records=records,
        meta={
            "simple_pass_rate_distribution": simple_pass_rate_distribution,
            "simple_mean_pass_rate": simple_mean_pass_rate,
            "strong_difficulty_score_distribution": strong_difficulty_score_distribution,
            "strong_difficulty_mean": strong_mean,
            "strong_failed_count": strong_failed_count,
        },
    )


def run_difficulty_eval(
    dataset: str,
    iter_n: int = 1,
    model: str = "*",
    root: str = "feedback_data/synth_clean",
    max_rows: int | None = None,
    output_root: str = "data_eval_output",
) -> EvalResult:
    df = load_synth_parquet(dataset, iter_n, model, root)
    root_path = Path(root) / dataset
    pattern = f"synthetic_iter{iter_n}_{model}.parquet"
    matched = sorted(f for f in root_path.glob(pattern) if not f.name.endswith("_hard.parquet"))
    file_stem = matched[0].stem if matched else f"synthetic_iter{iter_n}_{model}"
    return run_difficulty_eval_on_df(
        df=df,
        dataset=dataset,
        file_stem=file_stem,
        max_rows=max_rows,
        output_root=output_root,
    )
