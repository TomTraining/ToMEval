from __future__ import annotations

import hashlib
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from src.llm.client import LLMResponse

from .lang import get_sample_lang
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

{answer_instruction}"""

# 中文样本使用中文指令模板（参考 ToMBench 官方中文 prompt 风格）；
# 选项字母 A-Z 与 \boxed{} 输出协议保持不变，保证下游规则判分逻辑通用。
OPEN_QA_TEMPLATE_ZH = """请根据下面的故事回答问题。

故事：
{story}

问题：
{question}

请直接回答问题。
只输出答案文本。"""

CHOICE_QA_TEMPLATE_ZH = """请根据下面的故事回答问题。

故事：
{story}

问题：
{question}

选项：
{options_block}

{answer_instruction}"""

ANSWER_INSTRUCTIONS = {
    ("en", "mcq_single"): "Select the single best option.\nPut your final answer letter inside \\boxed{}, e.g. \\boxed{A}.",
    ("en", "mcq_multi"): "Select every correct option.\nPut all your final answer letters inside one \\boxed{}, comma-separated, e.g. \\boxed{A,C}.",
    ("zh", "mcq_single"): "请选出唯一最合适的选项。\n将最终答案的选项字母放进 \\boxed{} 中，例如 \\boxed{A}。",
    ("zh", "mcq_multi"): "请选出所有正确的选项。\n将所有最终答案的选项字母放进同一个 \\boxed{} 中，用英文逗号分隔，例如 \\boxed{A,C}。",
}


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
    # 指令语言跟随样本语言（meta.lang / meta.language），结构协议不变。
    lang = get_sample_lang(sample.get("meta"))
    if current_prompt_type == "open":
        template = OPEN_QA_TEMPLATE_ZH if lang == "zh" else OPEN_QA_TEMPLATE
        return template.format(story=sample["story"], question=sample["question"])

    # 在 prompt 中显式展示“字母 -> 文本”的当轮映射，方便 prediction 和 judge 对齐。
    options_block = "\n".join(f"{letter}. {text}" for letter, text in option_map.items())
    # 要求模型把最终答案放进 \boxed{}，metric 阶段通过 boxed 提取做规则判分。
    answer_instruction = ANSWER_INSTRUCTIONS[(lang, current_prompt_type)]
    template = CHOICE_QA_TEMPLATE_ZH if lang == "zh" else CHOICE_QA_TEMPLATE
    return template.format(
        story=sample["story"],
        question=sample["question"],
        options_block=options_block,
        answer_instruction=answer_instruction,
    )


def extract_boxed(text: str) -> Optional[str]:
    # 兼容 \boxed{...} 和 \box{...}；取最后一个匹配（最终答案通常在推理末尾）。
    matches = re.findall(r"\\box(?:ed)?\s*\{([^{}]*)\}", text)
    if not matches:
        return None
    return matches[-1]


def extract_prediction_value(current_prompt_type: PromptType, response: Optional[LLMResponse]) -> Any:
    if response is None or response.content is None:
        return None
    return extract_prediction_from_text(current_prompt_type, str(response.content))


def extract_prediction_from_text(current_prompt_type: PromptType, content: str) -> Any:
    # judge 阶段直接拿 prediction.jsonl 里的 content 文本做提取，不依赖 LLMResponse 对象。
    text = content.strip()
    if current_prompt_type == "open":
        return text

    # MCQ 严格模式：只认 \boxed{} 里的内容，没有 boxed 就视为提取失败（返回 None）。
    boxed = extract_boxed(text)
    if boxed is None:
        return None

    if current_prompt_type == "mcq_multi":
        # 从 boxed 内容提取字母列表并去重，天然兼容 "A,C" / "A, C" / "AC" 等写法。
        letters = []
        seen = set()
        for token in re.findall(r"[A-Za-z]", boxed):
            letter = token.upper()
            if letter not in seen:
                letters.append(letter)
                seen.add(letter)
        return letters or None

    m = re.search(r"[A-Za-z]", boxed)
    return m.group(0).upper() if m else None
