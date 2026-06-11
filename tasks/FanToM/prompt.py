"""FANToM 原论文 prompt(忠实复刻 skywalker023/fantom 的 eval_fantom.py 输入模板)。

官方各题型输入模板：
  factQA / belief:   {context}\\n\\nQuestion: {q}\\nAnswer:
  answerability:     {context}\\n\\nTarget: {fact_q}\\nQuestion: {q}\\nAnswer:
  info-accessibility:{context}\\n\\nInformation: {fact_q} {fact_a}\\nQuestion: {q}\\nAnswer:
  binary 题额外带 "Answer yes or no."
前置固定 prompt_header(theory-of-mind test …)。默认用 short_context(官方默认/最易档)。

Target/Information 前缀属上下文(保留)；"Answer:" 结尾与 "Answer yes or no." 属答案格式
(按统一策略移交 system prompt 的 \\boxed{} 指令)。fact_question/fact_answer/short_context
由 scripts/convert_fantom.py 补进 meta。
"""

from __future__ import annotations

from typing import Dict, Optional

from src.evaluation.lang import get_sample_lang
from src.evaluation.prompts import boxed_directive, prompt_type, render_options_block
from src.evaluation.protocols import reasoning_for
from src.evaluation.types import StandardizedSample

HEADER = (
    "This is a theory-of-mind test. Please answer the question regarding facts or beliefs, "
    "based on the following in-person conversation between individuals who have just met."
)

TEMPLATE = """{header}

{context}

{extra}Question: {question}

Options:
{options_block}"""


def _extra_line(meta: Dict) -> str:
    qtype = str(meta.get("question_type", ""))
    fact_q = meta.get("fact_question", "")
    fact_a = meta.get("fact_answer", "")
    if qtype.startswith("answerability"):
        return f"Target: {fact_q}\n"
    if qtype.startswith("infoAccessibility"):
        return f"Information: {fact_q} {fact_a}\n"
    return ""


def build_system_prompt(sample, protocol: str, lang: str, prompt_type: str) -> str:
    return boxed_directive(lang, prompt_type, reasoning_for(protocol))


def build_prompt(
    sample: StandardizedSample,
    option_map: Optional[Dict[str, str]],
    include_instruction: bool = True,
) -> str:
    meta = sample.get("meta") or {}
    lang = get_sample_lang(meta)
    context = meta.get("short_context") or sample["story"]
    body = TEMPLATE.format(
        header=HEADER,
        context=context,
        extra=_extra_line(meta),
        question=sample["question"],
        options_block=render_options_block(option_map or {}),
    )
    if include_instruction:
        body += "\n\n" + boxed_directive(lang, prompt_type(sample["answer"]), reasoning=False)
    return body.rstrip()
