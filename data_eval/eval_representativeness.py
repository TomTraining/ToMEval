"""F037：代表性评估（0-5 分 + 主维度 dimension_breakdown）。

强 LLM 对每条合成数据给 0-5 整数分（与声称的 ToM 维度的对齐程度），
并按 meta 主维度做 dimension_breakdown：{dim_value: {n, mean_score}}。

报告写到 data_eval_output/representativeness/<DS>_<stem>.json。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from data_eval.base import EvalResult, load_sample_rows, load_synth_parquet, write_report
from data_eval.prompts import DATASET_SKILL_REGISTRY, REPRESENTATIVENESS_PROMPTS  # noqa: F401
from src.llm.content_client import ContentClient
from src.llm.llm_utils import extract_json

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "config.yaml"

# F037 D037-02：数据集→主维度字段（用于 dimension_breakdown 分组）
_DIMENSION_FIELD: dict[str, str] = {
    "BigToM": "condition_type",
    "EmoBench": "subset",
    "FanToM": "question_type",
    "HiToM": "order",
    "SimpleToM": "dimension",
    "SocialIQA": "dimension",
    "ToMBench": "ability",
}

# 各数据集 prompt 所需 meta 字段（沿用 F020）
_META_FIELDS: dict[str, list[str]] = {
    "BigToM": ["condition_type", "dimension"],
    "EmoBench": ["subset", "question_subtype"],
    "FanToM": ["question_type", "order"],
    "HiToM": ["order"],
    "SimpleToM": ["dimension", "difficulty"],
    "SocialIQA": ["dimension"],
    "ToMBench": ["ability"],
}

# F035 对齐：固定采样 seed；F041：行数从 config.yaml sample_rows 读
_SAMPLE_SEED = 20260527


def _load_client() -> ContentClient:
    """F039：强模型 = config.yaml eval_model.strong（与 F035 强模型一致：deepseek-v4-flash）。"""
    from data_eval.base import load_answer_models
    return load_answer_models()["strong"]


def _meta_val_to_str(val: Any) -> str:
    import numpy as np
    if isinstance(val, (list, np.ndarray)):
        items = list(val)
        return items[0] if len(items) == 1 else ", ".join(str(v) for v in items)
    return str(val) if val is not None else ""


def _build_prompt(dataset: str, row: Any) -> str:
    template = REPRESENTATIVENESS_PROMPTS[dataset]
    answer = row["answer"] if isinstance(row["answer"], dict) else {}
    correct = answer.get("correct_answers", [])
    if isinstance(correct, (list, tuple)) and correct:
        correct_str = ", ".join(str(x) for x in correct)
    else:
        correct_str = str(correct)
    meta = row.get("meta") or {}
    kwargs: dict[str, Any] = {
        "story": str(row.get("story", "")),
        "question": str(row.get("question", "")),
        "correct_answers": correct_str,
    }
    for field in _META_FIELDS.get(dataset, []):
        kwargs[field] = _meta_val_to_str(meta.get(field, ""))
    return template.format(**kwargs)


def _parse_response(text: str | None) -> dict[str, Any]:
    """解析 {score:0-5, dim:str|null, reason:str}；越界/失败 → score=None。"""
    if not text:
        return {"score": None, "dim": None, "reason": ""}
    parsed = extract_json(text)
    if not parsed:
        return {"score": None, "dim": None, "reason": text[:200]}
    s = parsed.get("score")
    score: int | None
    if isinstance(s, (int, float)):
        si = int(s)
        score = si if 0 <= si <= 5 else None
    else:
        score = None
    dim = parsed.get("dim")
    if dim is not None and not isinstance(dim, str):
        dim = str(dim)
    reason = parsed.get("reason", "")
    return {"score": score, "dim": dim if dim else None, "reason": str(reason)}


def run_representativeness_eval_on_df(
    df: Any,
    dataset: str,
    file_stem: str,
    max_rows: int | None = None,
    output_root: str = "data_eval_output",
    output_subdir: str = "representativeness",
) -> EvalResult:
    """F038：对任意 df 跑 F037 代表性评估。供 run_train_eval.py 复用。"""
    if dataset not in REPRESENTATIVENESS_PROMPTS:
        raise ValueError(f"dataset={dataset!r} 不在 REPRESENTATIVENESS_PROMPTS 中")

    sample_size = max_rows if max_rows is not None else load_sample_rows(dataset)
    sample_size = min(sample_size, len(df))
    if sample_size < len(df):
        df = df.sample(n=sample_size, random_state=_SAMPLE_SEED).sort_index()
    else:
        df = df.copy()

    client = _load_client()

    prompts: list[str] = []
    row_indices: list[int] = []
    meta_ids: list[str] = []
    dim_field = _DIMENSION_FIELD.get(dataset, "dimension")
    dim_values: list[str] = []

    for idx, row in df.iterrows():
        prompts.append(_build_prompt(dataset, row))
        row_indices.append(int(idx))
        meta = row.get("meta") or {}
        meta_ids.append(str(meta.get("id", "")) if isinstance(meta, dict) else "")
        dim_values.append(_meta_val_to_str(meta.get(dim_field, "")) if isinstance(meta, dict) else "")

    responses = client.batch_generate(prompts, desc=f"representativeness/{dataset}/{file_stem}")

    score_distribution: dict[str, int] = {str(s): 0 for s in range(6)}
    parse_error_count = 0
    score_sum = 0
    scored_count = 0
    breakdown_raw: dict[str, list[int]] = {}
    records: list[dict[str, Any]] = []

    for i, resp in enumerate(responses):
        parsed = _parse_response(resp.content if resp else None)
        score = parsed["score"]
        dim_val = dim_values[i]
        if score is None:
            parse_error_count += 1
        else:
            score_distribution[str(score)] += 1
            score_sum += score
            scored_count += 1
            breakdown_raw.setdefault(dim_val, []).append(score)

        records.append({
            "row_idx": row_indices[i],
            "meta_id": meta_ids[i],
            "score": score,
            "inferred_dim": parsed["dim"],
            "reason": parsed["reason"],
            "claimed_dim_value": dim_val,
        })

    dimension_breakdown: dict[str, Any] = {}
    for dim, scores in breakdown_raw.items():
        n = len(scores)
        dimension_breakdown[dim] = {
            "n": n,
            "mean_score": round(sum(scores) / n, 3) if n > 0 else None,
        }

    mean_score = round(score_sum / scored_count, 3) if scored_count > 0 else None

    report: dict[str, Any] = {
        "dataset": dataset,
        "eval_type": "representativeness",
        "total_rows": len(df),
        "scored_count": scored_count,
        "parse_error_count": parse_error_count,
        "mean_representativeness_score": mean_score,
        "score_distribution": score_distribution,
        "dimension_breakdown": dimension_breakdown,
        "records": records,
    }

    out_path = Path(output_root) / output_subdir / f"{dataset}_{file_stem}.json"
    write_report(report, out_path)

    return EvalResult(
        dataset=dataset,
        eval_type="representativeness",
        total_rows=len(df),
        pass_=True,
        records=records,
        meta={
            "mean_representativeness_score": mean_score,
            "score_distribution": score_distribution,
            "dimension_breakdown": dimension_breakdown,
            "parse_error_count": parse_error_count,
        },
    )


def run_representativeness_eval(
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
    return run_representativeness_eval_on_df(
        df=df,
        dataset=dataset,
        file_stem=file_stem,
        max_rows=max_rows,
        output_root=output_root,
    )
