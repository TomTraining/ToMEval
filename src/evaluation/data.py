from __future__ import annotations

from typing import Any, Dict, List

import yaml

from src import runner

from .types import AnswerBlock, StandardizedSample


def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_task_config(config_path: str) -> Dict[str, Any]:
    config = read_yaml(config_path)
    if "dataset" not in config or "path" not in config:
        raise ValueError(f"{config_path} must contain 'dataset' and 'path'.")
    return config


def normalize_text_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def build_sample_id(row: Dict[str, Any], index: int) -> str:
    meta = row["meta"]
    return str(row.get("sample_id") or meta.get("id") or f"sample_{index}")


def normalize_sample(row: Dict[str, Any], index: int) -> StandardizedSample:
    answer = row.get("answer")
    meta = row.get("meta")
    if not isinstance(answer, dict):
        raise ValueError("Each standardized sample must contain an 'answer' object.")
    if not isinstance(meta, dict):
        raise ValueError("Each standardized sample must contain a 'meta' object.")

    story = row.get("story")
    question = row.get("question")
    if not isinstance(story, str):
        raise ValueError("Each standardized sample must contain a string 'story'.")
    if not isinstance(question, str):
        raise ValueError("Each standardized sample must contain a string 'question'.")

    normalized_answer: AnswerBlock = {
        "correct_answers": normalize_text_list(answer.get("correct_answers")),
        "wrong_answers": normalize_text_list(answer.get("wrong_answers")),
    }

    normalized: StandardizedSample = {
        "sample_id": build_sample_id({"meta": meta, **row}, index),
        "story": story.strip(),
        "question": question.strip(),
        "answer": normalized_answer,
        "meta": meta,
    }

    if not normalized["question"]:
        raise ValueError(f"Sample {normalized['sample_id']} is missing question.")
    if not normalized_answer["correct_answers"]:
        raise ValueError(f"Sample {normalized['sample_id']} is missing correct_answers.")
    return normalized


def load_standardized_data(
    dataset_config: Dict[str, Any],
    experiment_config: Dict[str, Any],
) -> List[StandardizedSample]:
    # 这里只读取已经标准化完成的数据，不再兼容原始异构字段。
    rows = runner.load_and_limit_data(
        subset=dataset_config["path"],
        datasets_root=experiment_config["normalized_datasets_path"],
        max_samples=experiment_config["max_samples"],
    )
    return [normalize_sample(row, index) for index, row in enumerate(rows)]


def analyze_question_types(samples: List[StandardizedSample]) -> Dict[str, Any]:
    """分析数据集的题型分布"""
    open_count = 0
    mcq_count = 0

    for sample in samples:
        if not sample["answer"]["wrong_answers"]:
            open_count += 1
        else:
            mcq_count += 1

    total = len(samples)
    if open_count == 0:
        question_type = "Multiple Choice"
    elif mcq_count == 0:
        question_type = "Open QA"
    else:
        question_type = f"Mixed (MCQ: {mcq_count}, Open: {open_count})"

    return {
        "total": total,
        "open_count": open_count,
        "mcq_count": mcq_count,
        "question_type": question_type
    }
