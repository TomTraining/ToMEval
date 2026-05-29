"""F034：format 评估改为从 _audit JSON 推导规则。

validate_dataframe 接收 FormatRules，不再依赖模块级硬编码白名单。
新增 failure rule：correct_count_out_of_range / wrong_count_out_of_range。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_eval.base import EvalResult, load_synth_parquet, write_report
from data_eval.format_rules import FormatRules, load_rules


REQUIRED_COLUMNS = ("story", "question", "answer", "meta")


def _is_listlike(x: Any) -> bool:
    return isinstance(x, (list, tuple, np.ndarray, pd.Series))


def _to_list(x: Any) -> list:
    if isinstance(x, np.ndarray):
        return x.tolist()
    return list(x)


def _classify_prompt_type(correct: list, wrong: list) -> str:
    if not wrong:
        return "open"
    if len(correct) > 1:
        return "mcq_multi"
    return "mcq_single"


def _meta_id(row: pd.Series) -> str:
    meta = row.get("meta")
    if isinstance(meta, dict):
        mid = meta.get("id")
        return str(mid) if mid is not None else ""
    return ""


def _record(failures: list[dict], rule: str, row_idx: int, meta_id: str, detail: str) -> None:
    failures.append({"rule": rule, "row_idx": int(row_idx), "meta_id": meta_id, "detail": detail})


def validate_dataframe(df: pd.DataFrame, rules: FormatRules) -> list[dict]:
    """对单个 (dataframe, rules) 跑全部格式规则，返回 failures 列表（空=PASS）。"""
    failures: list[dict] = []
    dataset = rules.dataset

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        _record(failures, "schema.required_columns", -1, "", f"missing columns: {missing_cols}")
        return failures

    seen_ids: dict[str, int] = {}

    for idx, row in df.iterrows():
        mid = _meta_id(row)

        story = row["story"]
        if not isinstance(story, str) or not story.strip():
            _record(failures, "field.story_nonempty", idx, mid,
                    f"story 不是非空 str(type={type(story).__name__})")

        question = row["question"]
        if not isinstance(question, str) or not question.strip():
            _record(failures, "field.question_nonempty", idx, mid,
                    f"question 不是非空 str(type={type(question).__name__})")

        answer = row["answer"]
        ans_ok = False
        correct_list: list = []
        wrong_list: list = []
        if not isinstance(answer, dict):
            _record(failures, "field.answer_dict", idx, mid,
                    f"answer 不是 dict(type={type(answer).__name__})")
        else:
            keys = set(answer.keys())
            if not {"correct_answers", "wrong_answers"}.issubset(keys):
                _record(failures, "field.answer_keys", idx, mid,
                        f"answer 缺键,实际 keys={sorted(keys)}")
            else:
                ca = answer["correct_answers"]
                wa = answer["wrong_answers"]
                if not _is_listlike(ca):
                    _record(failures, "field.correct_answers_listlike", idx, mid,
                            f"correct_answers 不是 list/array(type={type(ca).__name__})")
                elif not _is_listlike(wa):
                    _record(failures, "field.wrong_answers_listlike", idx, mid,
                            f"wrong_answers 不是 list/array(type={type(wa).__name__})")
                else:
                    correct_list = _to_list(ca)
                    wrong_list = _to_list(wa)
                    if len(correct_list) < 1:
                        _record(failures, "field.correct_answers_nonempty", idx, mid,
                                "correct_answers 长度=0，至少需要 1 条")
                    else:
                        for j, item in enumerate(correct_list):
                            if not str(item).strip():
                                _record(failures, "field.correct_answers_no_blank", idx, mid,
                                        f"correct_answers[{j}] strip 后为空")
                                break
                        for j, item in enumerate(wrong_list):
                            if not str(item).strip():
                                _record(failures, "field.wrong_answers_no_blank", idx, mid,
                                        f"wrong_answers[{j}] strip 后为空")
                                break
                        ans_ok = True

        meta = row["meta"]
        if not isinstance(meta, dict):
            _record(failures, "field.meta_dict", idx, mid,
                    f"meta 不是 dict(type={type(meta).__name__})")
        else:
            m_id = meta.get("id")
            if not isinstance(m_id, str) or not m_id.strip():
                _record(failures, "field.meta_id_nonempty", idx, mid,
                        f"meta.id 不是非空 str(value={m_id!r})")
            else:
                if m_id in seen_ids:
                    _record(failures, "field.meta_id_unique", idx, mid,
                            f"meta.id={m_id!r} 与 row_idx={seen_ids[m_id]} 重复")
                else:
                    seen_ids[m_id] = int(idx)
            for field_name in rules.meta_required:
                if field_name not in meta:
                    _record(failures, "meta.required_field_present", idx, mid,
                            f"meta 缺字段 {field_name!r}(dataset={dataset})")
                    continue
                v = meta[field_name]
                if _is_listlike(v):
                    items = _to_list(v)
                    if len(items) == 0:
                        _record(failures, "meta.required_field_nonempty", idx, mid,
                                f"meta.{field_name} 是空 list(dataset={dataset})")
                    else:
                        for j, item in enumerate(items):
                            if not str(item).strip():
                                _record(failures, "meta.required_field_nonempty", idx, mid,
                                        f"meta.{field_name}[{j}] strip 后为空(dataset={dataset})")
                                break
                else:
                    if not str(v).strip():
                        _record(failures, "meta.required_field_nonempty", idx, mid,
                                f"meta.{field_name} strip 后为空(value={v!r}, dataset={dataset})")

        if ans_ok:
            ptype = _classify_prompt_type(correct_list, wrong_list)
            if ptype not in rules.prompt_types:
                _record(failures, "prompt_type.whitelist", idx, mid,
                        f"prompt_type={ptype} 不在 {dataset} 白名单 {sorted(rules.prompt_types)}")
            n_correct = len(correct_list)
            n_wrong = len(wrong_list)
            if n_correct not in rules.correct_count_allowed:
                _record(failures, "correct_count_out_of_range", idx, mid,
                        f"correct_count={n_correct} 不在 {sorted(rules.correct_count_allowed)}(dataset={dataset})")
            if n_wrong not in rules.wrong_count_allowed:
                _record(failures, "wrong_count_out_of_range", idx, mid,
                        f"wrong_count={n_wrong} 不在 {sorted(rules.wrong_count_allowed)}(dataset={dataset})")

    return failures


def run_format_eval_on_df(
    df: pd.DataFrame,
    dataset: str,
    file_stem: str,
    parquet_path: str,
    output_root: str = "data_eval_output",
    audit_root: str = "data_eval/_audit",
    output_subdir: str = "format",
) -> EvalResult:
    """F038：对任意 df 跑格式校验，规则从 audit JSON 推导。供 run_train_eval.py 复用。"""
    rules = load_rules(dataset, audit_root=audit_root)
    failures = validate_dataframe(df, rules)
    passed = len(failures) == 0

    report = {
        "dataset": dataset,
        "parquet_path": parquet_path,
        "eval_type": "format",
        "total_rows": len(df),
        "pass": passed,
        "rules": {
            "prompt_types": sorted(rules.prompt_types),
            "correct_count_allowed": sorted(rules.correct_count_allowed),
            "wrong_count_allowed": sorted(rules.wrong_count_allowed),
            "meta_required": list(rules.meta_required),
        },
        "rule_names": [
            "schema.required_columns",
            "field.story_nonempty",
            "field.question_nonempty",
            "field.answer_dict",
            "field.answer_keys",
            "field.correct_answers_listlike",
            "field.wrong_answers_listlike",
            "field.correct_answers_nonempty",
            "field.correct_answers_no_blank",
            "field.wrong_answers_no_blank",
            "field.meta_dict",
            "field.meta_id_nonempty",
            "field.meta_id_unique",
            "meta.required_field_present",
            "meta.required_field_nonempty",
            "prompt_type.whitelist",
            "correct_count_out_of_range",
            "wrong_count_out_of_range",
        ],
        "failures": failures,
    }

    out_path = Path(output_root) / output_subdir / f"{dataset}_{file_stem}.json"
    write_report(report, out_path)

    return EvalResult(
        dataset=dataset,
        eval_type="format",
        total_rows=len(df),
        pass_=passed,
        records=failures,
        meta={"failures_count": len(failures)},
    )


def run_format_eval(
    dataset: str,
    iter_n: int = 1,
    model: str = "*",
    root: str = "feedback_data/synth_clean",
    max_rows: int | None = None,
    output_root: str = "data_eval_output",
    audit_root: str = "data_eval/_audit",
) -> EvalResult:
    """对单个数据集的 synth_clean parquet 跑格式校验，规则从 audit JSON 推导。"""
    df = load_synth_parquet(dataset, iter_n, model, root)

    root_path = Path(root) / dataset
    pattern = f"synthetic_iter{iter_n}_{model}.parquet"
    matched = sorted(f for f in root_path.glob(pattern) if not f.name.endswith("_hard.parquet"))
    file_stem = matched[0].stem if matched else f"synthetic_iter{iter_n}_{model}"

    if max_rows is not None:
        df = df.head(max_rows)

    return run_format_eval_on_df(
        df=df,
        dataset=dataset,
        file_stem=file_stem,
        parquet_path=str(root_path / f"{file_stem}.parquet"),
        output_root=output_root,
        audit_root=audit_root,
    )
