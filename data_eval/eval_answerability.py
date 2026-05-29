"""F036：answerability 严格三阶级联。

阶段 A — 读 F035 difficulty 中间产物，按 simple_correct 分桶：
  all_passed     : sum(simple_correct) == 5
  partial_failed : 1 <= sum < 5
  all_failed     : sum == 0  → 进入 B

阶段 B — 对 stage_a all_failed 的样本，强模型 repeat=3：
  ≥1 次正确 → 升级为 partial_failed
  仍 0 次正确 → 进入 C

阶段 C — 强模型质量打分 5 标签 (truly_hard / label_error / ambiguous /
contradictory_premise / missing_info)；解析失败计 parse_error 不计入分母。

answerability_score = 1 - (label_error + ambiguous + contradictory_premise + missing_info) / total_sampled
（truly_hard 不扣分；parse_error 不计入分母）

报告写到 data_eval_output/answerability/<DS>_<stem>.json。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from data_eval.answerability_core import (
    classify_prompt_type as _classify_prompt_type,
    get_correct_letters as _get_correct_letters,
    build_answer_prompt as _build_answer_prompt,
    check_correct_single as _check_correct_single,
    check_correct_multi as _check_correct_multi,
)
from data_eval.base import EvalResult, load_sample_rows, load_synth_parquet, write_report
from data_eval.eval_difficulty import (
    _SAMPLE_SEED,
    _load_strong_client,
)
from data_eval.prompts import ANSWERABILITY_QUALITY_PROMPT
from src.llm.llm_utils import extract_json

logger = logging.getLogger(__name__)

_STRONG_REPEAT = 3
_QUALITY_LABELS = (
    "truly_hard",
    "label_error",
    "ambiguous",
    "contradictory_premise",
    "missing_info",
)
_PROBLEM_LABELS = {"label_error", "ambiguous", "contradictory_premise", "missing_info"}


def _difficulty_artifact_path(dataset: str, file_stem: str, output_root: str) -> Path:
    return Path(output_root) / "difficulty" / f"{dataset}_{file_stem}.json"


def _load_difficulty_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"F036 需要 F035 难度评估中间产物，但未找到：{path}\n"
            f"请先运行 `python run_eval.py --eval difficulty --dataset <DS>` 生成"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"F035 产物 {path} records 为空或非 list")
    return records


def _build_quality_prompt(row: Any) -> str:
    answer = row["answer"] if isinstance(row.get("answer"), dict) else {}
    correct = answer.get("correct_answers", [])
    wrong = answer.get("wrong_answers", [])
    correct_str = ", ".join(str(x) for x in correct) if isinstance(correct, (list, tuple)) and len(correct) > 0 else str(correct)
    wrong_str = ", ".join(str(x) for x in wrong) if isinstance(wrong, (list, tuple)) and len(wrong) > 0 else str(wrong)
    return ANSWERABILITY_QUALITY_PROMPT.format(
        story=str(row.get("story", "")),
        question=str(row.get("question", "")),
        correct_answers=correct_str,
        wrong_answers=wrong_str,
    )


def _parse_quality(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    parsed = extract_json(text)
    if not parsed:
        return None
    label = parsed.get("label")
    if label not in _QUALITY_LABELS:
        return None
    score = parsed.get("score")
    if isinstance(score, (int, float)):
        s = int(score)
        if not (0 <= s <= 5):
            s = None
    else:
        s = None
    return {"label": label, "score": s, "reason": str(parsed.get("reason", ""))[:200]}


def run_answerability_eval_on_df(
    df: Any,
    dataset: str,
    file_stem: str,
    max_rows: int | None = None,
    output_root: str = "data_eval_output",
    difficulty_artifact_path: str | Path | None = None,
    output_subdir: str = "answerability",
    difficulty_subdir: str = "difficulty",
) -> EvalResult:
    """F038：对任意 df 跑 F036 三阶级联 answerability。供 run_train_eval.py 复用。"""
    sample_size = max_rows if max_rows is not None else load_sample_rows(dataset)
    sample_size = min(sample_size, len(df))
    if sample_size < len(df):
        df = df.sample(n=sample_size, random_state=_SAMPLE_SEED).sort_index()
    else:
        df = df.copy()

    art_path = (
        Path(difficulty_artifact_path)
        if difficulty_artifact_path is not None
        else Path(output_root) / difficulty_subdir / f"{dataset}_{file_stem}.json"
    )
    diff_records = _load_difficulty_records(art_path)

    df_indices = [int(i) for i in df.index.tolist()]
    diff_indices = [int(r["row_idx"]) for r in diff_records]
    if len(diff_records) != len(df) or diff_indices != df_indices:
        raise ValueError(
            f"F036 D036-01 断言失败：F035 records 与本次采样不一致。\n"
            f"  artifact={art_path}\n"
            f"  diff len={len(diff_records)} sampled len={len(df)}\n"
            f"  diff first/last row_idx={diff_indices[:3]}..{diff_indices[-3:]}\n"
            f"  sampled first/last row_idx={df_indices[:3]}..{df_indices[-3:]}\n"
            f"  请用同一 max_rows 先重跑 difficulty。"
        )

    rows_list = list(df.iterrows())

    stage_a_dist = {"all_passed": 0, "partial_failed": 0, "all_failed": 0}
    records: list[dict[str, Any]] = []
    all_failed_idx: list[int] = []

    for i, ((_, row), drec) in enumerate(zip(rows_list, diff_records)):
        sc = drec.get("simple_correct", [])
        if not isinstance(sc, list) or len(sc) != 5:
            raise ValueError(
                f"F035 records[{i}].simple_correct 必须为长度 5 的 list，得到 {sc!r}"
            )
        s_pass = sum(1 for x in sc if x)
        if s_pass == 5:
            bucket = "all_passed"
        elif s_pass == 0:
            bucket = "all_failed"
        else:
            bucket = "partial_failed"
        stage_a_dist[bucket] += 1

        meta = row.get("meta")
        mid = str(meta.get("id", "")) if isinstance(meta, dict) else ""
        rec = {
            "row_idx": int(drec["row_idx"]),
            "meta_id": mid,
            "simple_correct": list(sc),
            "simple_pass_rate": s_pass,
            "stage_a": bucket,
            "stage_b": None,
            "stage_b_correct_count": None,
            "stage_c": None,
        }
        records.append(rec)
        if bucket == "all_failed":
            all_failed_idx.append(i)

    stage_b_dist = {"upgraded_partial": 0, "still_failed": 0}
    still_failed_idx: list[int] = []

    if all_failed_idx:
        strong_client = _load_strong_client()
        prompts_b: list[str] = []
        prompt_types_b: list[str] = []
        correct_letters_b: list[list[str]] = []
        for i in all_failed_idx:
            _, row = rows_list[i]
            pt = _classify_prompt_type(row)
            cl = _get_correct_letters(row)
            prompt_types_b.append(pt)
            correct_letters_b.append(cl)
            prompts_b.append(_build_answer_prompt(pt, row))

        m = len(all_failed_idx)
        flat_b = []
        for k in range(_STRONG_REPEAT):
            flat_b.extend(prompts_b)
        resps_b = strong_client.batch_generate(
            flat_b, desc=f"answerability/stage_b/{dataset}/{file_stem}"
        )

        for j, i in enumerate(all_failed_idx):
            n_correct = 0
            pt = prompt_types_b[j]
            cl = correct_letters_b[j]
            for k in range(_STRONG_REPEAT):
                resp = resps_b[k * m + j]
                raw = resp.content if resp and resp.content else None
                if raw is None:
                    continue
                ok = _check_correct_multi(raw, cl) if pt == "mcq_multi" else _check_correct_single(raw, cl)
                if ok:
                    n_correct += 1
            records[i]["stage_b_correct_count"] = n_correct
            if n_correct >= 1:
                records[i]["stage_b"] = "upgraded_partial"
                stage_b_dist["upgraded_partial"] += 1
            else:
                records[i]["stage_b"] = "still_failed"
                stage_b_dist["still_failed"] += 1
                still_failed_idx.append(i)

    stage_c_label_dist: dict[str, int] = {lbl: 0 for lbl in _QUALITY_LABELS}
    stage_c_label_dist["parse_error"] = 0

    if still_failed_idx:
        strong_client_c = _load_strong_client()
        prompts_c = [_build_quality_prompt(rows_list[i][1]) for i in still_failed_idx]
        resps_c = strong_client_c.batch_generate(
            prompts_c, desc=f"answerability/stage_c/{dataset}/{file_stem}"
        )
        for j, i in enumerate(still_failed_idx):
            resp = resps_c[j]
            raw = resp.content if resp and resp.content else None
            parsed = _parse_quality(raw)
            if parsed is None:
                stage_c_label_dist["parse_error"] += 1
                records[i]["stage_c"] = {"label": "parse_error", "score": None, "reason": (raw or "")[:200]}
            else:
                stage_c_label_dist[parsed["label"]] += 1
                records[i]["stage_c"] = parsed

    total_sampled = len(records)
    n_problem = sum(stage_c_label_dist[lbl] for lbl in _PROBLEM_LABELS)
    answerability_score = round(1.0 - n_problem / total_sampled, 4) if total_sampled > 0 else None

    report: dict[str, Any] = {
        "dataset": dataset,
        "eval_type": "answerability",
        "total_rows": total_sampled,
        "stage_a_distribution": stage_a_dist,
        "stage_b_distribution": stage_b_dist,
        "stage_c_label_distribution": stage_c_label_dist,
        "answerability_score": answerability_score,
        "records": records,
    }

    out_path = Path(output_root) / output_subdir / f"{dataset}_{file_stem}.json"
    write_report(report, out_path)
    logger.info(
        f"[answerability] {dataset}/{file_stem} — total={total_sampled} "
        f"A={stage_a_dist} B={stage_b_dist} score={answerability_score}"
    )

    return EvalResult(
        dataset=dataset,
        eval_type="answerability",
        total_rows=total_sampled,
        pass_=True,
        records=records,
        meta={
            "stage_a_distribution": stage_a_dist,
            "stage_b_distribution": stage_b_dist,
            "stage_c_label_distribution": stage_c_label_dist,
            "answerability_score": answerability_score,
        },
    )


def run_answerability_eval(
    dataset: str,
    iter_n: int = 1,
    model: str = "*",
    root: str = "feedback_data/synth_clean",
    max_rows: int | None = None,
    output_root: str = "data_eval_output",
    difficulty_artifact_path: str | Path | None = None,
) -> EvalResult:
    """三阶级联 answerability 评估。"""
    df = load_synth_parquet(dataset, iter_n, model, root)

    root_path = Path(root) / dataset
    pattern = f"synthetic_iter{iter_n}_{model}.parquet"
    matched = sorted(f for f in root_path.glob(pattern) if not f.name.endswith("_hard.parquet"))
    file_stem = matched[0].stem if matched else f"synthetic_iter{iter_n}_{model}"

    return run_answerability_eval_on_df(
        df=df,
        dataset=dataset,
        file_stem=file_stem,
        max_rows=max_rows,
        output_root=output_root,
        difficulty_artifact_path=difficulty_artifact_path,
    )
