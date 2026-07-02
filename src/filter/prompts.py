"""V3 数据飞轮 Phase C answerability 判断提示词。

V3 简化设计：
- 只保留 ANSWERABILITY_FULL_PROMPT（Phase C 使用）
- 修复提示词在 repair/repair_prompts.py
- 旧版本的 difficulty/representativeness 评估已废弃
"""

# ── V3 Phase C：answerability 全量单条 judge ──────────────────────────────────
# 对 partial + all_failed 样本判断是否可回答
# 输出：{"label": "answerable|label_error|ambiguous|contradictory_premise|missing_info", "reason": "..."}
# 决策树规则：answerable → 继续 Phase D；其余 4 类 → label=bad, repair_type=unanswerable

ANSWERABILITY_FULL_PROMPT = """\
You are auditing a Theory-of-Mind (ToM) question to decide whether it is logically answerable \
given only the story and question. Do NOT try to answer it; only judge whether the question is well-posed.

Story: {story}
Question: {question}
Correct answer: {correct_answers}
Wrong answers (distractors): {wrong_answers}

Choose EXACTLY ONE label:
  - answerable            : the story uniquely supports the marked correct answer; distractors are clearly wrong.
  - label_error           : the marked correct answer is wrong, or one of the distractors is actually correct.
  - ambiguous             : multiple options are equally defensible given the story.
  - contradictory_premise : the story contains an internal contradiction that blocks any consistent answer.
  - missing_info          : the story does not contain enough information to derive the correct answer.

Return ONLY valid JSON (no markdown fences):
{{
  "label": "<one of: answerable | label_error | ambiguous | contradictory_premise | missing_info>",
  "reason": "<one short sentence>"
}}"""


# 中文样本版本：指令中文化，但 label 仍输出英文枚举，保证下游决策树逻辑不变。
ANSWERABILITY_FULL_PROMPT_ZH = """\
你正在审核一道心智理论（Theory-of-Mind, ToM）题目，判断仅凭故事和问题该题在逻辑上是否可回答。\
不要尝试作答，只判断题目本身是否构造良好。

故事：{story}
问题：{question}
正确答案：{correct_answers}
错误答案（干扰项）：{wrong_answers}

请从以下标签中选择且只选择一个（label 必须输出英文原词）：
  - answerable            ：故事能唯一支持标注的正确答案，干扰项明显错误。
  - label_error           ：标注的正确答案是错的，或某个干扰项其实是对的。
  - ambiguous             ：根据故事，多个选项同样合理。
  - contradictory_premise ：故事存在内部矛盾，导致无法得出一致答案。
  - missing_info          ：故事缺少推出正确答案所需的信息。

只返回合法 JSON（不要 markdown 代码块）：
{{
  "label": "<one of: answerable | label_error | ambiguous | contradictory_premise | missing_info>",
  "reason": "<一句简短的中文理由>"
}}"""


__all__ = ["ANSWERABILITY_FULL_PROMPT", "ANSWERABILITY_FULL_PROMPT_ZH"]
