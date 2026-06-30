"""评测协议(protocol)定义:采样参数、system prompt 风格、答案 extractor。

四个协议(采样参数以下表为权威,等价于历史的 sampling_params_for_v4):

| protocol     | temperature | top_p | max_tokens | enable_thinking | n_samples |
|--------------|-------------|-------|------------|-----------------|-----------|
| direct       | 0.0         | 0.95  | 32768      | False           | 1         |
| direct_think | 0.6         | 0.95  | 32768      | True            | 1         |
| cot          | 0.6         | 0.95  | 32768      | True            | 1         |
| del_tom      | 0.6         | 0.95  | 32768      | True            | 8         |

协议还决定:
- system prompt 风格:direct/direct_think 走"裸答"(直接输出最终答案、不解释);
  cot/del_tom 走"step by step"(逐步推理、最终 \\boxed{} 放最后一行)。
- extractor:direct 用 extract_direct(第一个 \\boxed{});其余用 extract_cot(最后一个 \\boxed{})。
- del_tom 复用 repeats 跑 n_samples=8 次 + 多数投票,且不 shuffle。

本模块为纯函数,不做任何 I/O,且不依赖 prompts.py(extract_* 在此定义,由 prompts.py 反向导入)。
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from .types import PromptType

PROTOCOLS = ("direct", "direct_think", "cot", "del_tom")

# 与历史 sampling_params_for_v4 一致的权威采样参数表。
_SAMPLING: Dict[str, Dict[str, Any]] = {
    "direct": dict(temperature=0.0, top_p=0.95, max_tokens=32768, n_samples=1, enable_thinking=False),
    "direct_think": dict(temperature=0.6, top_p=0.95, max_tokens=32768, n_samples=1, enable_thinking=True),
    "cot": dict(temperature=0.6, top_p=0.95, max_tokens=32768, n_samples=1, enable_thinking=True),
    "del_tom": dict(temperature=0.6, top_p=0.95, max_tokens=32768, n_samples=8, enable_thinking=True),
}

# 复用与 prompts.extract_boxed 相同的正则(此处独立定义以避免与 prompts.py 循环导入)。
_BOXED_RE = r"\\box(?:ed)?\s*\{([^{}]*)\}"


def _check(protocol: str) -> None:
    if protocol not in _SAMPLING:
        raise ValueError(f"Unknown protocol: {protocol!r}; expected one of {PROTOCOLS}")


def sampling_params_for_v4(protocol: str) -> Dict[str, Any]:
    """返回协议的完整采样参数(副本)。"""
    _check(protocol)
    return dict(_SAMPLING[protocol])


def llm_overrides_for(protocol: str) -> Dict[str, Any]:
    """返回要覆盖到 llm_config 上的采样参数(不含 n_samples)。"""
    params = sampling_params_for_v4(protocol)
    return {
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "max_tokens": params["max_tokens"],
        "enable_thinking": params["enable_thinking"],
    }


def repeats_for(protocol: Optional[str], config_repeats: int) -> int:
    """设了协议→repeats=协议的 n_samples;未设→沿用 config 的 repeats。"""
    if protocol is None:
        return config_repeats
    return sampling_params_for_v4(protocol)["n_samples"]


def shuffle_for(protocol: Optional[str]) -> bool:
    """del_tom 不 shuffle(8 次选项顺序一致,按字母投票才有意义);其余 shuffle。"""
    return protocol != "del_tom"


def is_voting_protocol(protocol: Optional[str]) -> bool:
    return protocol == "del_tom"


def reasoning_for(protocol: Optional[str]) -> bool:
    """cot / del_tom 为推理型(可先逐步推理再给答案);direct / direct_think 为直答型。

    供 per-dataset 的 build_system_prompt 决定答题指令是否带“先推理”措辞。
    """
    return protocol in ("cot", "del_tom")


def _style(protocol: str) -> str:
    """direct/direct_think → 'bare';cot/del_tom → 'reason'。"""
    return "bare" if protocol in ("direct", "direct_think") else "reason"


def extractor_name_for(protocol: Optional[str]) -> str:
    """direct→'direct';direct_think/cot/del_tom→'cot';未设协议→'legacy'。"""
    if protocol is None:
        return "legacy"
    return "direct" if protocol == "direct" else "cot"


# system prompt:按 (风格 style, 语言 lang, 题型 ptype) 给出完整文案。
# 字母格式措辞与 prompts.ANSWER_INSTRUCTIONS 保持一致,保证下游 \boxed{} 解析通用。
_SYSTEM_PROMPTS: Dict[tuple, str] = {
    # ---------- bare(direct / direct_think):给出最终答案即可 ----------
    # 不再硬性禁止解释/推理：direct_think 开启了 thinking，禁推理会与之打架；
    # direct(不思考、temp=0)在此指令下也会稳定直接给出 \boxed 答案。
    ("bare", "en", "mcq_single"): (
        "You are a careful reader answering a multiple-choice theory-of-mind question. "
        "Read the story and the question carefully, then give your final answer in the "
        "format \\boxed{X} where X is the letter of the single best option."
    ),
    ("bare", "en", "mcq_multi"): (
        "You are a careful reader answering a multiple-choice theory-of-mind question. "
        "Read the story and the question carefully, then give your final answer as one "
        "\\boxed{} containing every correct option letter, comma-separated, e.g. \\boxed{A,C}."
    ),
    ("bare", "en", "open"): (
        "You are a careful reader answering a question about a story. "
        "Read the story and the question carefully, then give your final answer text."
    ),
    ("bare", "zh", "mcq_single"): (
        "你是一个认真阅读的人,正在回答一道关于心理状态(theory-of-mind)的单选题。"
        "请仔细阅读故事和问题,然后给出最终答案,格式为 \\boxed{X},其中 X 是唯一最合适选项的字母。"
    ),
    ("bare", "zh", "mcq_multi"): (
        "你是一个认真阅读的人,正在回答一道关于心理状态(theory-of-mind)的多选题。"
        "请仔细阅读故事和问题,然后给出最终答案:把所有正确选项的字母放进同一个 \\boxed{} 中,"
        "用英文逗号分隔,例如 \\boxed{A,C}。"
    ),
    ("bare", "zh", "open"): (
        "你是一个认真阅读的人,正在回答一道关于故事的问题。"
        "请仔细阅读故事和问题,然后给出最终的答案文本。"
    ),
    # ---------- reason(cot / del_tom):逐步推理,最终答案放最后一行 ----------
    ("reason", "en", "mcq_single"): (
        "You are a careful reader answering a multiple-choice theory-of-mind question. "
        "Think step by step about the mental states of the characters, then output your final "
        "answer in the format \\boxed{X} where X is the letter of the single best option. "
        "Put your final \\boxed{X} on the last line."
    ),
    ("reason", "en", "mcq_multi"): (
        "You are a careful reader answering a multiple-choice theory-of-mind question. "
        "Think step by step about the mental states of the characters, then output your final "
        "answer as one \\boxed{} containing every correct option letter, comma-separated, "
        "e.g. \\boxed{A,C}. Put your final \\boxed{} on the last line."
    ),
    ("reason", "en", "open"): (
        "You are a careful reader answering a question about a story. "
        "Think step by step about the mental states of the characters, then give your final answer. "
        "Put your final answer on the last line."
    ),
    ("reason", "zh", "mcq_single"): (
        "你是一个认真阅读的人,正在回答一道关于心理状态(theory-of-mind)的单选题。"
        "请一步步推理角色的心理状态,然后输出最终答案,格式为 \\boxed{X},其中 X 是唯一最合适选项的字母。"
        "把最终的 \\boxed{X} 放在最后一行。"
    ),
    ("reason", "zh", "mcq_multi"): (
        "你是一个认真阅读的人,正在回答一道关于心理状态(theory-of-mind)的多选题。"
        "请一步步推理角色的心理状态,然后输出最终答案:把所有正确选项的字母放进同一个 \\boxed{} 中,"
        "用英文逗号分隔,例如 \\boxed{A,C}。把最终的 \\boxed{} 放在最后一行。"
    ),
    ("reason", "zh", "open"): (
        "你是一个认真阅读的人,正在回答一道关于故事的问题。"
        "请一步步推理角色的心理状态,然后给出最终答案。把最终答案放在最后一行。"
    ),
    # ---------- mcq_grouped:一个故事下多道子问题,每问各输出一个 \boxed{} ----------
    ("bare", "en", "mcq_grouped"): (
        "You are answering several multiple-choice theory-of-mind questions about one story. "
        "For EACH question, in the given order, output your answer as \\boxed{X} where X is the "
        "letter of the single best option. Output exactly one \\boxed{} per question, in order, "
        "and nothing else (no explanation)."
    ),
    ("reason", "en", "mcq_grouped"): (
        "You are answering several multiple-choice theory-of-mind questions about one story. "
        "Think step by step about the mental states of the characters, then for EACH question, in "
        "the given order, output your final answer as \\boxed{X} (the letter of the single best "
        "option). Put one \\boxed{} per question, in order, on the last lines."
    ),
    ("bare", "zh", "mcq_grouped"): (
        "你是一个认真阅读的人,正在回答同一个故事下的多道单选题。"
        "请按给定顺序,对每一道子问题各输出一个 \\boxed{X},其中 X 是该问唯一最合适选项的字母。"
        "每问恰好一个 \\boxed{},按顺序排列,不要包含任何解释或多余文字。"
    ),
    ("reason", "zh", "mcq_grouped"): (
        "你是一个认真阅读的人,正在回答同一个故事下的多道单选题。"
        "请一步步推理角色的心理状态,然后按给定顺序,对每一道子问题各输出一个 \\boxed{X}"
        "(该问唯一最合适选项的字母)。把每问一个、按顺序排列的 \\boxed{} 放在最后几行。"
    ),
}


def system_prompt_for(protocol: str, lang: str, ptype: PromptType) -> str:
    """按 协议风格 × 语言 × 题型 返回 system prompt。"""
    _check(protocol)
    style = _style(protocol)
    lang = "zh" if str(lang).lower().startswith("zh") else "en"
    return _SYSTEM_PROMPTS[(style, lang, ptype)]


def extract_direct(text: str) -> Optional[str]:
    """direct 协议:取第一个 \\boxed{} 内容;没有则取全文第一个字母。"""
    matches = re.findall(_BOXED_RE, text)
    if matches:
        return matches[0]
    m = re.search(r"[A-Za-z]", text)
    return m.group(0) if m else None


def extract_cot(text: str) -> Optional[str]:
    """cot 协议:取最后一个 \\boxed{} 内容;没有则取末 200 字符内最后一个字母。"""
    matches = re.findall(_BOXED_RE, text)
    if matches:
        return matches[-1]
    tail_letters = re.findall(r"[A-Za-z]", text[-200:])
    return tail_letters[-1] if tail_letters else None


__all__ = [
    "PROTOCOLS",
    "sampling_params_for_v4",
    "llm_overrides_for",
    "repeats_for",
    "shuffle_for",
    "is_voting_protocol",
    "reasoning_for",
    "extractor_name_for",
    "system_prompt_for",
    "extract_direct",
    "extract_cot",
]
