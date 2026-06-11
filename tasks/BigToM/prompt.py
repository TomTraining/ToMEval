"""BigToM 原论文 prompt(忠实复刻 cicl-stanford/procedural-evals-tom)。

- system prompt 复刻官方 evaluate.txt / evaluate_cot.txt 的 "Answer the questions based on
  the context …"，把官方 "Answer as 'Answer:<option>)<answer>'" 输出格式换成本框架 \\boxed{}。
- body 复刻官方 "Story: … / Question: … / Choose one of the following: …"。
  官方用小写 a)/b)，这里用统一的字母 + \\boxed{} 协议。BigToM 为英文数据集。
"""

from __future__ import annotations

from typing import Dict, Optional

from src.evaluation.lang import get_sample_lang
from src.evaluation.prompts import boxed_directive, prompt_type, render_options_block
from src.evaluation.protocols import reasoning_for
from src.evaluation.types import StandardizedSample

BODY = """Story: {story}
Question: {question}
Choose one of the following:
{options_block}"""


def build_system_prompt(sample, protocol: str, lang: str, prompt_type: str) -> str:
    reasoning = reasoning_for(protocol)
    return "Answer the question based on the context. " + boxed_directive(lang, prompt_type, reasoning)


def build_prompt(
    sample: StandardizedSample,
    option_map: Optional[Dict[str, str]],
    include_instruction: bool = True,
) -> str:
    lang = get_sample_lang(sample.get("meta"))
    options_block = render_options_block(option_map or {})
    body = BODY.format(story=sample["story"], question=sample["question"], options_block=options_block)
    if include_instruction:
        body += "\n\n" + boxed_directive(lang, prompt_type(sample["answer"]), reasoning=False)
    return body.rstrip()
