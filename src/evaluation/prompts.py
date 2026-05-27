from __future__ import annotations

import hashlib
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from src.llm.client import LLMResponse

from .types import AnswerBlock, PromptType, StandardizedSample


OPTION_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

OPEN_QA_TEMPLATE = """You are answering a question about a story.

Story:
{story}

Question:
{question}

Answer the question directly.
Return only the answer text."""

CHOICE_QA_TEMPLATE = """You are answering a question about a story.

Story:
{story}

Question:
{question}

Options:
{options_block}

{answer_instruction}
Return only the answer."""


def build_option_bundle(
    dataset_name: str,
    sample_id: str,
    answer: AnswerBlock,
    repeat: int,
) -> Tuple[Optional[Dict[str, str]], List[str], List[str], int]:
    # 没有 wrong_answers 就视为开放题，不生成选项映射。
    correct_answers = list(answer["correct_answers"])
    wrong_answers = list(answer["wrong_answers"])
    if not wrong_answers:
        return None, [], [], 0

    option_rows = [{"text": text, "is_correct": True} for text in correct_answers] + [
        {"text": text, "is_correct": False} for text in wrong_answers
    ]
    if len(option_rows) > len(OPTION_LETTERS):
        raise ValueError(f"Too many options for sample {sample_id}: {len(option_rows)}")

    # shuffle 必须可复现，所以只依赖 dataset/sample/repeat 生成确定性种子。
    seed_source = f"{dataset_name}|{sample_id}|{repeat}"
    seed = int(hashlib.sha256(seed_source.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    rng.shuffle(option_rows)

    letters = list(OPTION_LETTERS[: len(option_rows)])
    option_map = {letter: row["text"] for letter, row in zip(letters, option_rows)}
    correct_letters = [letter for letter, row in zip(letters, option_rows) if row["is_correct"]]
    wrong_letters = [letter for letter, row in zip(letters, option_rows) if not row["is_correct"]]
    return option_map, correct_letters, wrong_letters, seed


def prompt_type(answer: AnswerBlock) -> PromptType:
    if not answer["wrong_answers"]:
        return "open"
    if len(answer["correct_answers"]) > 1:
        return "mcq_multi"
    return "mcq_single"


def build_prompt(sample: StandardizedSample, option_map: Optional[Dict[str, str]]) -> str:
    current_prompt_type = prompt_type(sample["answer"])
    if current_prompt_type == "open":
        return OPEN_QA_TEMPLATE.format(story=sample["story"], question=sample["question"])

    # 在 prompt 中显式展示“字母 -> 文本”的当轮映射，方便 prediction 和 judge 对齐。
    options_block = "\n".join(f"{letter}. {text}" for letter, text in option_map.items())
    answer_instruction = (
        "Select every correct option and return a list of option letters."
        if current_prompt_type == "mcq_multi"
        else "Select the single best option and return exactly one option letter."
    )
    return CHOICE_QA_TEMPLATE.format(
        story=sample["story"],
        question=sample["question"],
        options_block=options_block,
        answer_instruction=answer_instruction,
    )


def extract_prediction_value(current_prompt_type: PromptType, response: Optional[LLMResponse]) -> Any:
    if response is None or response.content is None:
        return None
    text = str(response.content).strip()
    if current_prompt_type == "open":
        return text
    if current_prompt_type == "mcq_multi":
        # 多选题统一提取字母列表并去重，避免模型输出 "A, A, C" 这类噪声。
        letters = []
        seen = set()
        for token in re.findall(r"[A-Za-z]", text):
            letter = token.upper()
            if letter not in seen:
                letters.append(letter)
                seen.add(letter)
        return letters

    # 优先匹配 "答案是 X" / "Answer: X" / "option X" / 独立大写字母等模式
    # 依次尝试更精确的模式，最后才回退到首字母
    for pattern in (
        r"(?:answer|option|选项|答案)[^\w]*([A-Z])\b",
        r"\b([A-Z])\b",
        r"([A-Za-z])",
    ):
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return ""
