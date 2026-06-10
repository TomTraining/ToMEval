from __future__ import annotations

import json
from typing import Any, Dict, List

from .lang import get_sample_lang
from .prompts import extract_prediction_from_text
from .storage import pred_content


def backfill_meta(result: Dict[str, Any], record: Dict[str, Any]) -> None:
    """把样本身份(sample_id / repeat / prompt_type)回填进判分结果。

    判分函数只产出 is_correct / error_reason 等，metric 聚合还需要样本身份；
    且协议模式下题型指令在 system prompt，仅凭 prompt 字段看不出题型，这里显式带上。
    judge_repeat 和 voting.vote_collapse 共用本 helper。
    """
    result.setdefault("sample_id", record["sample_id"])
    result.setdefault("repeat", record["repeat"])
    result.setdefault("prompt_type", record["prompt_type"])


def judge_prompt(record: Dict[str, Any]) -> str:
    # 只有 open 题走 LLM judge，MCQ 一律走 rule_judge_mcq 规则判分。
    assert record["prompt_type"] == "open", f"judge_prompt only supports open, got {record['prompt_type']}"

    # 始终使用模型的完整原始输出，避免提取失败影响正确率。
    model_output = pred_content(record) or ""

    # storyless 版本：不再把 story / question 输入给 judge，
    # 只给 correct / wrong 答案 + 模型预测，要求 judge 做"参照式"对比判定。
    correct_answers: List[str] = record.get("correct_answers") or []
    wrong_answers: List[str] = record.get("wrong_answers") or []

    # 中文样本用中文措辞的 judge prompt，JSON 输出契约（{"is_correct": ...}）不变。
    if get_sample_lang(record.get("meta")) == "zh":
        return f"""你需要把模型的回答与参考答案对比，判定其是否正确。

可接受的正确答案：
{json.dumps(correct_answers, ensure_ascii=False)}

已知的错误答案（不能匹配这些）：
{json.dumps(wrong_answers, ensure_ascii=False)}

模型回答：
{model_output}

只输出一个 JSON 对象：{{"is_correct": true}} 或 {{"is_correct": false}}
仅当模型回答在语义上至少匹配一个正确答案时，is_correct 才为 true。
如果模型回答匹配了已知错误答案、与正确答案矛盾或与问题无关，is_correct 为 false。
允许轻微的措辞差异。"""

    return f"""You are grading a QA response by comparing it against reference answers.

Accepted correct answers:
{json.dumps(correct_answers, ensure_ascii=False)}

Known wrong answers (must NOT match these):
{json.dumps(wrong_answers, ensure_ascii=False)}

Model response:
{model_output}

Output ONLY a JSON object: {{"is_correct": true}} or {{"is_correct": false}}
Mark is_correct as true ONLY if the model response semantically matches at least one accepted correct answer.
Mark is_correct as false if the model response matches a known wrong answer, contradicts the correct answers, or is irrelevant.
Minor wording differences are acceptable."""


def rule_judge_mcq(record: Dict[str, Any]) -> Dict[str, Any]:
    # MCQ 规则判分：从模型原始输出中提取 \boxed{} 答案，和正确字母直接比对。
    model_output = pred_content(record)
    if model_output in (None, ""):
        return {"is_correct": False, "error_reason": "content_none", "extracted": None}

    # extractor 跟随记录里 stamp 的协议（direct/cot/legacy），单独跑 stage=metric 也能正确分派。
    extractor = record.get("extractor", "legacy")
    extracted = extract_prediction_from_text(record["prompt_type"], str(model_output), extractor)
    if extracted is None:
        # 严格模式：没有 \boxed{} 或 boxed 内无字母，直接判错并标记提取失败。
        return {"is_correct": False, "error_reason": "extraction_failed", "extracted": None}

    correct_letters: List[str] = record.get("correct_letters") or []
    if record["prompt_type"] == "mcq_multi":
        is_correct = set(extracted) == set(correct_letters)
    else:
        is_correct = extracted in correct_letters
    return {
        "is_correct": is_correct,
        "error_reason": None if is_correct else "wrong_answer",
        "extracted": extracted,
    }


def judge_repeat(
    records: List[Dict[str, Any]],
    judge_client: Any = None,
    *,
    open_ctx: Any = None,
) -> List[Dict[str, Any]]:
    """对一批记录判分:MCQ 走规则提取,open 题按 open_ctx 选定的模式判分。

    open_ctx:OpenJudgeContext(由 open_judge.build_open_ctx 构建)。缺省时退回用
    judge_client 构造 llm_simple 上下文(ctx_from_judge_client),便于只传 judge_client
    的简单调用方。
    """
    from .open_judge import ctx_from_judge_client, judge_open

    if open_ctx is None:
        open_ctx = ctx_from_judge_client(judge_client)

    # 先给每个样本放一个默认错误结果，后面只覆盖真正拿到判分结果的样本。
    per_sample_results: List[Dict[str, Any]] = [
        {"is_correct": False, "error_reason": "content_none"} for _ in records
    ]

    open_records: List[Dict[str, Any]] = []
    open_indices: List[int] = []
    for index, record in enumerate(records):
        # MCQ 走规则判分，不消耗 judge API。
        if record["prompt_type"] in ("mcq_single", "mcq_multi"):
            per_sample_results[index] = rule_judge_mcq(record)
            continue
        open_records.append(record)
        open_indices.append(index)

    # open 题统一交给 open_judge 分派(f1 / llm_simple / rubric)。
    for index, result in zip(open_indices, judge_open(open_records, open_ctx)):
        per_sample_results[index] = result

    for record, result in zip(records, per_sample_results):
        backfill_meta(result, record)
    return per_sample_results
