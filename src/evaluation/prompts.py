from __future__ import annotations

import hashlib
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from src.llm.client import LLMResponse

from .lang import get_sample_lang
from .protocols import extract_cot, extract_direct
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
    shuffle: bool = True,
) -> Tuple[Optional[Dict[str, str]], List[str], List[str], int]:
    # 只有“无干扰项且唯一正确答案”才是开放题，不生成选项映射；
    # 无 wrong 但多正确（如 5选5）仍按 MCQ 用正确项构造选项，避免被误判为开放题。
    correct_answers = list(answer["correct_answers"])
    wrong_answers = list(answer["wrong_answers"])
    if not wrong_answers and len(correct_answers) == 1:
        return None, [], [], 0

    option_rows = [{"text": text, "is_correct": True} for text in correct_answers] + [
        {"text": text, "is_correct": False} for text in wrong_answers
    ]
    if len(option_rows) > len(OPTION_LETTERS):
        raise ValueError(f"Too many options for sample {sample_id}: {len(option_rows)}")

    # shuffle 必须可复现，所以只依赖 dataset/sample/repeat 生成确定性种子。
    # del_tom 等投票协议关闭 shuffle，保证同一 sample 各 repeat 选项顺序一致，按字母投票才有意义。
    seed_source = f"{dataset_name}|{sample_id}|{repeat}"
    seed = int(hashlib.sha256(seed_source.encode("utf-8")).hexdigest()[:16], 16)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(option_rows)

    letters = list(OPTION_LETTERS[: len(option_rows)])
    option_map = {letter: row["text"] for letter, row in zip(letters, option_rows)}
    correct_letters = [letter for letter, row in zip(letters, option_rows) if row["is_correct"]]
    wrong_letters = [letter for letter, row in zip(letters, option_rows) if not row["is_correct"]]
    return option_map, correct_letters, wrong_letters, seed


def prompt_type(answer: AnswerBlock) -> PromptType:
    # 开放题严格定义:无干扰项且正确答案恰好 1 个。
    # 防止“无 wrong 但多正确”(如 5选5)被误判为开放题——这类按多选题处理。
    if not answer["wrong_answers"] and len(answer["correct_answers"]) == 1:
        return "open"
    if len(answer["correct_answers"]) > 1:
        return "mcq_multi"
    return "mcq_single"


# 把官方各自的答案格式(ToMBench [[X]] / EmoBench JSON / BigToM Answer:<option> / 生成式)
# 统一替换成本框架的 \boxed{} 协议。reasoning=True 时措辞允许“先推理再给答案”。
# 供 per-dataset 的 build_system_prompt 复用，保证下游 \boxed 抽取通用。
_BOXED_DIRECTIVE = {
    ("en", "mcq_single", False): "Give your answer as \\boxed{X}, where X is the letter of the single best option.",
    ("en", "mcq_single", True): "Reason step by step about the characters' mental states, then give your final answer as \\boxed{X}, where X is the letter of the single best option.",
    ("en", "mcq_multi", False): "Give your answer as one \\boxed{} containing every correct option letter, comma-separated, e.g. \\boxed{A,C}.",
    ("en", "mcq_multi", True): "Reason step by step, then give your final answer as one \\boxed{} containing every correct option letter, comma-separated, e.g. \\boxed{A,C}.",
    ("en", "mcq_grouped", False): "For each question in order, give its answer as a separate \\boxed{X}, e.g. \\boxed{A} for question 1 then \\boxed{C} for question 2.",
    ("en", "mcq_grouped", True): "Reason step by step, then for each question in order give its answer as a separate \\boxed{X}, e.g. \\boxed{A} for question 1 then \\boxed{C} for question 2.",
    ("en", "open", False): "Give only your final answer text.",
    ("en", "open", True): "Reason step by step, then give your final answer text.",
    ("zh", "mcq_single", False): "请把答案放进 \\boxed{X},其中 X 是唯一最合适选项的字母。",
    ("zh", "mcq_single", True): "请先一步步推理角色的心理状态,然后把最终答案放进 \\boxed{X},其中 X 是唯一最合适选项的字母。",
    ("zh", "mcq_multi", False): "请把所有正确选项的字母放进同一个 \\boxed{} 中,用英文逗号分隔,例如 \\boxed{A,C}。",
    ("zh", "mcq_multi", True): "请先一步步推理,然后把所有正确选项的字母放进同一个 \\boxed{} 中,用英文逗号分隔,例如 \\boxed{A,C}。",
    ("zh", "mcq_grouped", False): "请按顺序对每一道问题各输出一个 \\boxed{X},例如第一问 \\boxed{A}、第二问 \\boxed{C}。",
    ("zh", "mcq_grouped", True): "请先一步步推理,然后按顺序对每一道问题各输出一个 \\boxed{X},例如第一问 \\boxed{A}、第二问 \\boxed{C}。",
    ("zh", "open", False): "只输出最终答案文本。",
    ("zh", "open", True): "请先一步步推理,然后给出最终答案文本。",
}


def boxed_directive(lang: str, current_prompt_type: PromptType, reasoning: bool) -> str:
    """返回 \\boxed{} 答题格式指令(替换官方各自的答案格式)。"""
    lang = "zh" if str(lang).lower().startswith("zh") else "en"
    return _BOXED_DIRECTIVE[(lang, current_prompt_type, reasoning)]


def render_options_block(option_map: Dict[str, str]) -> str:
    """渲染“字母. 文本”选项块；per-dataset prompt.py 复用，保证与默认模板一致。"""
    return "\n".join(f"{letter}. {text}" for letter, text in option_map.items())


def answer_instruction_for(lang: str, current_prompt_type: PromptType, include_instruction: bool) -> str:
    """协议评测(include_instruction=False)时答题格式指令由 system prompt 承载，body 留空。"""
    if not include_instruction:
        return ""
    return ANSWER_INSTRUCTIONS[(lang, current_prompt_type)]


def build_prompt(
    sample: StandardizedSample,
    option_map: Optional[Dict[str, str]],
    include_instruction: bool = True,
) -> str:
    current_prompt_type = prompt_type(sample["answer"])
    # 指令语言跟随样本语言（meta.lang / meta.language），结构协议不变。
    lang = get_sample_lang(sample.get("meta"))
    if current_prompt_type == "open":
        template = OPEN_QA_TEMPLATE_ZH if lang == "zh" else OPEN_QA_TEMPLATE
        return template.format(story=sample["story"], question=sample["question"])

    # 在 prompt 中显式展示“字母 -> 文本”的当轮映射，方便 prediction 和 judge 对齐。
    options_block = "\n".join(f"{letter}. {text}" for letter, text in option_map.items())
    # include_instruction=False 时（协议评测），答题格式指令改由 system prompt 承载，user prompt 只留选项块。
    answer_instruction = ANSWER_INSTRUCTIONS[(lang, current_prompt_type)] if include_instruction else ""
    template = CHOICE_QA_TEMPLATE_ZH if lang == "zh" else CHOICE_QA_TEMPLATE
    return template.format(
        story=sample["story"],
        question=sample["question"],
        options_block=options_block,
        answer_instruction=answer_instruction,
    ).rstrip()


def extract_boxed(text: str) -> Optional[str]:
    # 兼容 \boxed{...} 和 \box{...}；取最后一个匹配（最终答案通常在推理末尾）。
    matches = re.findall(r"\\box(?:ed)?\s*\{([^{}]*)\}", text)
    if not matches:
        return None
    return matches[-1]


def extract_prediction_value(
    current_prompt_type: PromptType,
    response: Optional[LLMResponse],
    extractor: str = "legacy",
) -> Any:
    if response is None or response.content is None:
        return None
    return extract_prediction_from_text(current_prompt_type, str(response.content), extractor)


def extract_prediction_from_text(
    current_prompt_type: PromptType,
    content: str,
    extractor: str = "legacy",
) -> Any:
    # judge 阶段直接拿 prediction.jsonl 里的 content 文本做提取，不依赖 LLMResponse 对象。
    text = content.strip()
    if current_prompt_type == "open":
        return text

    # 协议感知的 boxed 提取：direct 取第一个 \boxed{}，cot 取最后一个，legacy 沿用历史的“最后一个”。
    # 三者均无 boxed 时返回 None（MCQ 严格模式：视为提取失败）。
    if extractor == "direct":
        boxed = extract_direct(text)
    elif extractor == "cot":
        boxed = extract_cot(text)
    else:
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
