"""
数据合成Prompt模板

包含：
1. 批量维度诊断 prompt（K 条 bad case → DimensionDiagnosisReport）
2. 从诊断报告合成数据的 prompt（DimensionDiagnosisReport → 新样本）
3. 推理链方法 prompt（7种理论框架，模型自主选择）
"""

import json
from typing import Any, Dict, List, Optional


# ============ 数据集格式注册表 ============
# 每个数据集的格式规范，用于合成时约束输出格式

SYNTHESIS_FORMAT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "ToMBench": {
        "story_structure": "string",
        "options_in_question": True,
        "n_correct": 1,
        "n_wrong": None,
        "answer_key": "Correct Answer",
        "schema": "MCQAnswer",
        "meta_ability_field": "ability",
        "meta_required_fields": ["ability", "id", "lang", "qa_index"],
        "description": "4-choice MCQ where options are embedded in the Question text. Answer is a single letter (A/B/C/D).",
        "complexity_hint": (
            "Story: 2-4 sentences (100-400 words), 2-4 named agents, everyday social scenario. "
            "Covers: false beliefs, intentions, non-literal communication (faux pas/irony/white lies), second-order beliefs. "
            "Style: natural English prose, use Western-style names. "
            "Question: asks about one agent's mental state, belief, or intention."
        ),
        "json_skeleton": {
            "questions": [{
                "story": "<complete story text>",
                "question": "<question with A. B. C. D. options embedded>",
                "correct_letter": "A",
                "meta_ability": "theory_of_mind",
                "meta_id": "synthetic_0001",
            }]
        },
    },
    "SocialIQA": {
        "story_structure": "dict_full_story",
        "options_in_question": False,
        "n_correct": 1,
        "n_wrong": 2,
        "answer_key": "Correct_Answer",
        "schema": "MCQAnswer3",
        "meta_ability_field": "dimension",
        "meta_required_fields": ["id", "dataset_source", "dimension", "task_type", "difficulty"],
        "description": "3-choice MCQ. Story is a dict with full_story field. Provide exactly 1 Correct_Answer and 2 Wrong_Answer.",
        "complexity_hint": (
            "SIMPLE everyday scenario. Story: 1-3 short sentences (20-80 words), typically 1-2 agents. "
            "NO formal institution names, NO multi-step information chains. "
            "Question: asks what someone would do/want/feel/react next (xWant/xNeed/xReact/xEffect/xAttr/oWant/oReact/oEffect). "
            "Style: casual commonsense, NOT belief-tracking or ToM. "
            "Wrong answers: plausible everyday alternatives, not absurd."
        ),
        "json_skeleton": {
            "questions": [{
                "story_full": "<story text>",
                "question": "<question text>",
                "correct_answer": "<correct answer>",
                "wrong_answer_1": "<wrong answer 1>",
                "wrong_answer_2": "<wrong answer 2>",
                "meta_id": "synthetic_0001",
                "meta_dimension": "xWant",
                "meta_difficulty": "medium",
            }]
        },
    },
    "BigToM": {
        "story_structure": "dict_full_story",
        "options_in_question": False,
        "n_correct": 1,
        "n_wrong": 1,
        "answer_key": "Correct_Answer",
        "schema": "MCQAnswer2",
        "meta_ability_field": "condition_type",
        "meta_required_fields": ["id", "dataset_source", "condition_type", "dimension"],
        "description": (
            "Binary choice. Story involves an agent's belief state. Provide 1 Correct_Answer and 1 Wrong_Answer. "
            "meta_condition_type MUST match the diagnosed condition type from the report: "
            "forward_belief / backward_belief / forward_action / percept_to_belief / true_belief / false_belief."
        ),
        "complexity_hint": (
            "Story: 3-6 sentences, exactly 2 agents, clear belief-state scenario. "
            "The condition_type determines the question structure:\n"
            "  - false_belief: agent A is absent when object-state changes; ask what A believes\n"
            "  - true_belief: agent A witnesses the change; ask what A believes (correct)\n"
            "  - forward_belief: given events, what will agent believe next\n"
            "  - backward_belief: given agent's current belief, what event caused it\n"
            "  - forward_action: given agent's belief, what action will they take\n"
            "  - percept_to_belief: given agent's perception, what do they believe\n"
            "Wrong answer: what a model outputs when it ignores the agent's perspective and uses world-state truth."
        ),
        "json_skeleton": {
            "questions": [{
                "story_full": "<story text>",
                "question": "<question about belief/action/perception>",
                "correct_answer": "<correct answer from agent's perspective>",
                "wrong_answer": "<wrong answer — uses world-state truth instead of agent's belief>",
                "meta_id": "synthetic_001__backward_belief",
                "meta_condition_type": "backward_belief",
                "meta_dimension": "first_order",
            }]
        },
    },
    "EmoBench": {
        "story_structure": "dict_full_story",
        "options_in_question": False,
        "n_correct": 1,
        "n_wrong": "variable",
        "answer_key": "Correct_Answer",
        "schema": "OpenAnswer",
        "meta_ability_field": "coarse_category",
        "meta_required_fields": ["id", "subset", "language", "question_subtype", "coarse_category", "finegrained_category", "dimension", "choice_texts"],
        "description": "Emotion reasoning. Story is dict with full_story. Answer uses text match (not letter). Correct_Answer is the exact emotion text.",
        "complexity_hint": (
            "Story: 2-5 sentences (50-200 words), 1-2 agents. "
            "Describes a situation with a clear emotional trigger (goal blocked/achieved, social event, loss/gain). "
            "Question: asks what emotion the agent feels (cause/reaction/desire). "
            "Style: natural emotional narrative. "
            "Correct answer and wrong answers are ALL single emotion words (e.g. 'joy', 'embarrassment', 'relief'). "
            "Wrong answers must be emotions that are plausible but incorrect given the specific situation."
        ),
        "json_skeleton": {
            "questions": [{
                "story_full": "<story text>",
                "question": "<emotion question>",
                "correct_answer": "<correct emotion text>",
                "wrong_answer_1": "<wrong emotion 1>",
                "wrong_answer_2": "<wrong emotion 2>",
                "wrong_answer_3": "",
                "meta_id": "synthetic_0001",
                "meta_question_subtype": "cause",
                "meta_coarse_category": "<emotion category>",
                "meta_finegrained_category": "<specific emotion>",
                "meta_dimension": "emotion_attribution",
            }]
        },
    },
    "FanToM": {
        "story_structure": "dict_full_story",
        "options_in_question": False,
        "n_correct": 1,
        "n_wrong": 1,
        "answer_key": "Correct_Answer",
        "schema": "MCQAnswer2",
        "meta_ability_field": "question_type",
        "meta_required_fields": ["id", "question_type"],
        "description": (
            "Belief QA in multi-party conversation context. Story is dict with full_story and summary. "
            "question_type must be one of: beliefQAs, factQA, answerabilityQAs_binary, infoAccessibilityQAs_binary. "
            "Provide 1 Correct_Answer and 1 Wrong_Answer."
        ),
        "complexity_hint": (
            "Story: multi-party conversation (3-6 participants), 200-600 words. "
            "Some participants join or leave mid-conversation, creating partial information access. "
            "Question types: beliefQAs (what does X believe), factQA (factual recall), "
            "answerabilityQAs_binary (can X answer question Y), infoAccessibilityQAs_binary (did X hear info Z). "
            "Style: realistic dialogue with clear speaker turns. "
            "The key challenge: participants differ in what parts of the conversation they witnessed."
        ),
        "json_skeleton": {
            "questions": [{
                "story_full": "<full conversation story>",
                "story_summary": "<brief summary>",
                "question": "<question about what someone believes>",
                "correct_answer": "<correct answer>",
                "wrong_answer": "<wrong answer>",
                "meta_id": "synthetic_001__beliefQAs__1",
                "meta_question_type": "beliefQAs",
            }]
        },
    },
    "HiToM": {
        "story_structure": "dict_full_story",
        "options_in_question": False,
        "n_correct": 1,
        "n_wrong": 14,
        "answer_key": "Correct_Answer",
        "schema": "MCQAnswer15",
        "meta_ability_field": "order",
        "meta_required_fields": ["id", "order"],
        "description": "15-choice MCQ testing higher-order ToM. Story is dict with full_story. Provide exactly 1 correct_answer and 14 wrong options (wrong_1 through wrong_14). meta_order indicates ToM depth (1-4).",
        "complexity_hint": (
            "Story: exactly 5 named agents, 10-20 numbered action steps (e.g. '1. Avery enters the room.'). "
            "Each step records who is present and who observes each object move. "
            "Object is moved multiple times; each agent's belief depends on which steps they witnessed. "
            "meta_order (1-4) = recursion depth of the belief question. "
            "Style: structured numbered narrative, NOT prose. "
            "15 answer options = all possible locations/states the object could be in from any agent's perspective."
        ),
        "json_skeleton": {
            "questions": [{
                "story_full": "<story with multiple agents>",
                "question": "<higher-order belief question>",
                "correct_answer": "<correct answer>",
                "wrong_1": "<wrong option 1>",
                "wrong_2": "<wrong option 2>",
                "wrong_3": "<wrong option 3>",
                "wrong_4": "<wrong option 4>",
                "wrong_5": "<wrong option 5>",
                "wrong_6": "<wrong option 6>",
                "wrong_7": "<wrong option 7>",
                "wrong_8": "<wrong option 8>",
                "wrong_9": "<wrong option 9>",
                "wrong_10": "<wrong option 10>",
                "wrong_11": "<wrong option 11>",
                "wrong_12": "<wrong option 12>",
                "wrong_13": "<wrong option 13>",
                "wrong_14": "<wrong option 14>",
                "meta_id": "synthetic_0001",
                "meta_order": 2,
            }]
        },
    },
    "SimpleToM": {
        "story_structure": "dict_full_story",
        "options_in_question": False,
        "n_correct": 1,
        "n_wrong": 2,
        "answer_key": "Correct_Answer",
        "schema": "MCQAnswer3",
        "meta_ability_field": "dimension",
        "meta_required_fields": ["id", "dataset_source", "dimension", "difficulty"],
        "description": "3-choice MCQ. One wrong answer is always 'Empty' (representing N/A). Story can have full_story + summary + background fields. Provide 1 correct_answer and 1 wrong_answer (the third option 'Empty' is added automatically).",
        "complexity_hint": (
            "Story: 3-8 sentences (80-250 words), 2-3 agents, everyday social situations. "
            "Covers: behavior prediction (what will X do), mental state awareness (what does X know/think), "
            "moral judgment (was X's action appropriate). "
            "Style: natural prose, clear causal chain. "
            "Wrong answer: a plausible but incorrect prediction or judgment. "
            "Note: a third option 'Empty' (meaning none/N/A) is added automatically — your wrong_answer should NOT be the 'Empty' option."
        ),
        "json_skeleton": {
            "questions": [{
                "story_full": "<story text>",
                "story_summary": "",
                "story_background": "",
                "question": "<social inference question>",
                "correct_answer": "<correct answer>",
                "wrong_answer": "<wrong answer>",
                "meta_id": "synthetic_0001",
                "meta_dimension": "social_exchange",
                "meta_difficulty": "medium",
            }]
        },
    },
}


# ============ 数据集技能说明注册表 ============

DATASET_SKILL_REGISTRY: Dict[str, str] = {
    "SocialIQA": (
        "Commonsense social reasoning: what people do/want/feel/react in everyday situations. "
        "NOT belief-tracking or ToM. Focuses on single-agent causal/motivational inference "
        "(xWant, xNeed, xReact, xEffect, xAttr, oWant, oReact, oEffect)."
    ),
    "ToMBench": (
        "Multi-dimensional Theory of Mind: false beliefs, intentions, non-literal communication "
        "(faux pas, irony, white lies), second-order beliefs, knowledge attribution."
    ),
    "BigToM": (
        "First-order false belief vs true belief. Binary question: does Agent A believe "
        "object is in state X or state Y? Agent A's belief depends on whether they witnessed "
        "the state change."
    ),
    "EmoBench": (
        "Emotion attribution: identify the specific emotion an agent feels given a situation "
        "and their goal. Covers cause (why does X feel Y), reaction (how does X feel about Z), "
        "desire (what does X want after Y)."
    ),
    "SimpleToM": (
        "Social behavior prediction and moral judgment in simple everyday scenarios. "
        "1-2 agents, covers: what will X do next, what does X know/think, "
        "was X's action socially appropriate."
    ),
    "HiToM": (
        "Higher-order recursive belief (order 1-4): A thinks B thinks C thinks D thinks... "
        "5 agents, each with partial observation of object movements. "
        "Must track nested belief chains across multiple steps."
    ),
    "FanToM": (
        "Belief tracking in multi-party conversation: who-knows-what after partial "
        "information sharing. Participants join/leave mid-conversation, creating "
        "asymmetric information access."
    ),
}


# ============ Stage 2: 批量维度诊断 Prompt ============

BATCH_DIMENSION_DIAGNOSIS_PROMPT = """You are an expert in Theory of Mind (ToM) cognitive science and systematic reasoning failure analysis.

A language model failed on the following {K} questions from the **{dimension}** dimension of the **{dataset_name}** dataset.

## Dataset Cognitive Skill
{dataset_skill}

## Failed Cases ({K} instances from the same dimension)

{bad_cases_section}

## Your Task: Cross-Case Pattern Analysis

Analyze these {K} failures AS A BATCH to identify the STRUCTURAL ERROR PATTERN that explains why the model systematically failed — not the specific content of any single question.

Key principle: Patterns appearing in 2+ cases reveal systematic cognitive weaknesses worth targeting for data synthesis. Single-case errors may be noise.

## Output Requirements
1. Use ONLY abstract placeholders: "Agent A/B/C", "Object X/Y", "Location X/Y" — ZERO original names/content from the questions above
2. Describe COGNITIVE OPERATIONS and structural patterns, not surface features
3. recommended_synthesis_themes: general story settings or contexts (NOT specific question content)
4. difficulty_distribution values must sum to 1.0

cognitive_operation options (choose the most specific match):
  belief_state_update    — failed to update an agent's belief after an unwitnessed event
  perspective_taking     — failed to simulate a specific agent's viewpoint
  emotion_attribution    — failed to infer an agent's emotional state from a situation
  recursive_belief       — failed to track nested belief chains (A thinks B thinks C...)
  information_access     — failed to track who has access to which information
  social_inference       — failed to infer social norms, intentions, or behavioral implications
  causal_reasoning       — failed to trace cause-effect chains in a narrative
  temporal_reference     — failed to resolve which event a temporal phrase refers to
  pragmatic_inference    — failed to infer implied meaning beyond literal text
  referent_resolution    — failed to resolve what a pronoun or deictic term refers to
  common_sense_normative — failed to apply social norms or appropriateness standards

Return ONLY the JSON (no markdown):
{{
  "dimension": "{dimension}",
  "common_error_patterns": ["<abstract pattern using Agent A/B/Object X/Location X placeholders>", ...],
  "primary_cognitive_operation": "<one of the 11 options above>",
  "secondary_cognitive_operations": ["<optional secondary op>", ...],
  "recommended_synthesis_themes": ["<story theme/setting 1>", "<story theme/setting 2>", "<story theme/setting 3>"],
  "difficulty_distribution": {{"easy": 0.2, "medium": 0.5, "hard": 0.3}}
}}"""


def build_batch_diagnosis_prompt(
    bad_cases_batch: List[Dict[str, Any]],
    dimension: str,
    dataset_name: str,
) -> str:
    """构建批量维度诊断 prompt

    K 条来自同一维度的错误样本 → DimensionDiagnosisReport
    """
    dataset_skill = DATASET_SKILL_REGISTRY.get(dataset_name, "General Theory of Mind reasoning.")
    K = len(bad_cases_batch)

    cases_sections = []
    for i, case in enumerate(bad_cases_batch, 1):
        actual_prompt = str(case.get("_actual_prompt", "")).strip()
        wrong_reasoning = str(case.get("_wrong_reasoning", "")).strip()
        wrong_answer = str(case.get("_wrong_answer", "")).strip()
        gold = str(case.get("_gold_answer", "")).strip()

        if len(actual_prompt) > 1500:
            actual_prompt = actual_prompt[:1300] + "\n...[truncated]"
        if wrong_reasoning and len(wrong_reasoning) > 800:
            wrong_reasoning = wrong_reasoning[:700] + "\n...[truncated]"

        section = f"### Case {i}\n**Prompt**:\n{actual_prompt or '(not available)'}\n\n"
        if wrong_reasoning:
            section += f"**Model Reasoning**:\n{wrong_reasoning}\n\n"
        section += f"**Model Answer**: {wrong_answer or '(unknown)'}\n**Correct Answer**: {gold or '(unknown)'}\n"
        cases_sections.append(section)

    bad_cases_section = "\n---\n".join(cases_sections)

    return BATCH_DIMENSION_DIAGNOSIS_PROMPT.format(
        K=K,
        dimension=dimension,
        dataset_name=dataset_name,
        dataset_skill=dataset_skill,
        bad_cases_section=bad_cases_section,
    )


# ============ Stage 3: 从诊断报告合成数据 Prompt ============

# 干扰项设计原则（按认知操作类型）
_DISTRACTOR_STRATEGY_BY_OPERATION: Dict[str, str] = {
    "belief_state_update": (
        "Primary trap: use the REAL-WORLD current state as the main wrong answer. "
        "A model that fails to update the agent's belief (and instead outputs world truth) will pick this. "
        "Secondary trap (if multiple wrong answers): use a third agent's belief state."
    ),
    "perspective_taking": (
        "Primary trap: use what IS actually true from an omniscient view as a wrong answer. "
        "A model that can't take the specific agent's viewpoint defaults to the objective truth. "
        "Secondary trap: use what a different agent (not the queried one) believes."
    ),
    "information_access": (
        "Primary trap: use the ground truth (omniscient/full-context answer) as a wrong answer — "
        "a model that doesn't filter by who-knows-what will confidently pick this. "
        "Secondary trap: use what a well-informed agent who DID have access would answer."
    ),
    "recursive_belief": (
        "Primary trap: answer one level shallower than asked "
        "(e.g., for 'what does A think B believes', trap = what B actually believes, not A's model of B). "
        "Secondary trap: use the ground truth."
    ),
    "emotion_attribution": (
        "Primary trap: use an emotion with the SAME valence but from a DIFFERENT trigger or context "
        "(e.g., if 'grief' is correct due to loss, 'disappointment-from-failure' is a trap). "
        "Secondary trap: use an emotion with opposite valence that ignores the key situational detail."
    ),
    "social_inference": (
        "Primary trap: use the superficially logical or literal action (ignoring social norms/implications). "
        "A model that doesn't infer social context defaults to the surface-level 'obvious' action."
    ),
    "temporal_reference": (
        "Primary trap: answer 'what to do before/after the WRONG sub-event' — "
        "pick the low-level enabling step for the subsidiary action X rather than the prerequisite for main event Y. "
        "Secondary trap: use a semantically adjacent but chronologically displaced step."
    ),
    "pragmatic_inference": (
        "Primary trap: the LITERAL / face-value interpretation of the utterance as a wrong answer. "
        "A model that doesn't infer implicature, hints, or indirect meaning picks the literal reading."
    ),
    "causal_reasoning": (
        "Primary trap: reverse the cause-effect direction (swap cause and effect) as a wrong answer. "
        "Secondary trap: use a plausible intermediate step that is not the actual cause."
    ),
    "referent_resolution": (
        "Primary trap: use a different syntactically plausible referent for the pronoun/deictic term. "
        "Make sure multiple referents are genuinely ambiguous from the surface form."
    ),
    "common_sense_normative": (
        "Primary trap: use a response that is factually accurate but socially inappropriate in context. "
        "Secondary trap: a response that seems correct in isolation but violates the specific social norm being tested."
    ),
}

_DIFFICULTY_RULES: Dict[str, str] = {
    "easy": (
        "Easy difficulty rules:\n"
        "- Story: 1-3 sentences, minimal setup, clear and direct narrative\n"
        "- At most 2 agents, no nested information chains\n"
        "- Wrong options: must be the answer a model gives when it SKIPS the target cognitive operation. "
        "Do NOT use obviously absurd or unrelated distractors."
    ),
    "medium": (
        "Medium difficulty rules:\n"
        "- Story: 3-6 sentences, 2-3 agents, one key information gap or misdirection\n"
        "- Wrong options: at least one must exploit the specific cognitive shortcut described in DISTRACTOR DESIGN. "
        "At least one distractor should require reading two-thirds of the story to realize it is wrong."
    ),
    "hard": (
        "Hard difficulty rules:\n"
        "- Story: 3+ agents with overlapping but distinct knowledge states\n"
        "- Multiple sequential events where only some agents are present\n"
        "- At least one deliberate misdirection: surface-level clues point toward a wrong answer\n"
        "- Wrong options: at least one matches the REAL-WORLD truth (trapping agents who ignore perspective), "
        "at least one matches what a different agent would believe"
    ),
}

STAGE2_GENERATION_FROM_REPORT_PROMPT = """You are an expert at creating Theory of Mind (ToM) evaluation questions that consistently challenge language models.

## Dataset Format Specification
{format_description}

## Dataset Story Style (MUST FOLLOW)
{complexity_hint}

## Required JSON Structure
{json_skeleton}

## Dimension-Level Error Pattern Analysis
{report_summary}

The new question MUST target the following cognitive weakness:
**Primary Cognitive Operation**: {primary_cognitive_operation}

**Common Error Patterns (cross-case analysis)**:
{common_error_patterns}

**Recommended Story Themes/Contexts**:
{synthesis_themes}

## Complexity Specification (MUST FOLLOW — do NOT override)
- Target difficulty: **{difficulty}**
- Story style: follow the Dataset Story Style section above exactly

{difficulty_specific_rules}

## DISTRACTOR DESIGN [CRITICAL — this is why questions currently fail to challenge models]
Wrong answers must NOT be randomly wrong or obviously absurd. Each wrong answer must be
the answer a model produces when it SKIPS or SHORTCUTS the required cognitive operation.

Distractor strategy for **{primary_cognitive_operation}**:
{distractor_strategy}

Self-check every wrong answer before finalizing:
1. Would a model that skips {primary_cognitive_operation} plausibly pick this wrong answer? → must be YES
2. Is this wrong answer obviously, trivially incorrect? → must be NO
3. Does picking the CORRECT answer specifically require performing {primary_cognitive_operation}? → must be YES

If any check fails, revise the wrong answer until all three pass.

## [COGNITIVE OPERATION LOCK — HIGHEST PRIORITY]
The question you generate MUST test EXACTLY this cognitive operation: **{primary_cognitive_operation}**
Before submitting, verify: "Does solving this question specifically require {primary_cognitive_operation}?"
If the question accidentally tests a different cognitive operation, revise it.

## Generation Rules
0. [COGNITIVE OPERATION LOCK] See above — cognitive operation must be exactly {primary_cognitive_operation}
1. **LANGUAGE: Write ALL text (story, question, answers) in ENGLISH ONLY. No Chinese, no other languages.**
2. Create a completely ORIGINAL question — all character names, locations, objects, and scenarios must be fresh
3. Do NOT reuse names like Alice/Bob/Maria/John or scenarios from known benchmarks
4. Strictly follow the JSON structure and story style shown above
5. Choose a story theme from the Recommended Story Themes if possible
6. {format_specific_rules}
7. Set meta_difficulty to "{difficulty}" (from the Complexity Specification above)
8. After drafting, verify: (a) cognitive operation is {primary_cognitive_operation}, (b) story length matches the Dataset Story Style, (c) all wrong answers pass the DISTRACTOR DESIGN self-check

{retry_feedback}## Output
Generate {target_count} question(s) following the exact format above.
Return ONLY the JSON:

{{
  "questions": [
    <question 1 following the required structure>,
    ...
  ]
}}"""


def build_stage2_generation_from_report_prompt(
    report: Dict[str, Any],
    dataset_name: str,
    target_count: int = 1,
    previous_question: Optional[Dict[str, Any]] = None,
) -> str:
    """构建从维度诊断报告合成新样本的 prompt

    取代旧的 build_stage2_generation_prompt（后者基于逐条 ErrorDiagnosisOutput）。
    新方法基于 DimensionDiagnosisReport，每份报告代表一个维度的系统性弱点。

    Args:
        report: DimensionDiagnosisReport dict
        dataset_name: 数据集名称
        target_count: 生成数量
        previous_question: 上一轮生成但未通过验证的题目（retry feedback）
    """
    fmt = SYNTHESIS_FORMAT_REGISTRY.get(dataset_name)
    if fmt is None:
        fmt = SYNTHESIS_FORMAT_REGISTRY["SocialIQA"]

    # 从 difficulty_distribution 中选最高概率的难度
    difficulty_distribution = report.get("difficulty_distribution", {})
    if difficulty_distribution:
        difficulty = max(difficulty_distribution, key=lambda k: difficulty_distribution.get(k, 0))
    else:
        difficulty = "medium"

    primary_op = str(report.get("primary_cognitive_operation", "belief_state_update"))

    n_wrong = fmt["n_wrong"]
    if n_wrong == 0:
        format_specific_rules = "This is open-ended — do NOT include Wrong_Answer field."
    elif n_wrong == "variable":
        format_specific_rules = "Put all answer options (including correct) in Meta.choice_texts. Correct_Answer must exactly match one of the choice_texts values."
    elif n_wrong is None:
        format_specific_rules = "Embed all answer options (A/B/C/D) inside the Question text."
    else:
        format_specific_rules = f"Provide exactly {fmt['n_correct']} Correct_Answer and {n_wrong} Wrong_Answer item(s)."

    difficulty_specific_rules = _DIFFICULTY_RULES.get(difficulty, _DIFFICULTY_RULES["medium"])
    distractor_strategy = _DISTRACTOR_STRATEGY_BY_OPERATION.get(
        primary_op,
        "Wrong answers must be the answer a model gives when it shortcuts the required cognitive operation. "
        "Do NOT use obviously absurd or unrelated distractors.",
    )

    if previous_question is not None:
        prev_str = json.dumps(previous_question, indent=2, ensure_ascii=False)
        retry_feedback = (
            "## ⚠ RETRY — Previous Attempt Was Too Easy\n"
            "The question below was generated for this same error pattern but rejected because "
            "the target model answered it correctly every time. "
            "Study what made it too easy, then generate a HARDER version.\n\n"
            "### Previous Question (rejected)\n"
            f"```json\n{prev_str}\n```\n\n"
            "### How to make it harder\n"
            "- Add more surface-level misdirection in the story that points toward a wrong answer\n"
            "- Make at least one wrong answer represent EXACTLY what the model outputs when it "
            f"skips {primary_op} — it must be irresistible to a model that shortcuts\n"
            "- Use a completely different story scenario (new names, setting, situation)\n\n"
        )
    else:
        retry_feedback = ""

    common_patterns = report.get("common_error_patterns", [])
    synthesis_themes = report.get("recommended_synthesis_themes", [])
    dimension = report.get("dimension", "unknown")

    # 对 BigToM：dimension 字段即 condition_type，在 format_specific_rules 里额外强调
    if dataset_name == "BigToM" and dimension and dimension != "unknown":
        format_specific_rules = (
            f"{format_specific_rules} "
            f"IMPORTANT: set meta_condition_type to exactly \"{dimension}\" "
            f"(this is the condition type diagnosed from the bad cases)."
        )

    report_summary = json.dumps({
        "dimension": dimension,
        "common_error_patterns": common_patterns,
        "primary_cognitive_operation": primary_op,
        "secondary_cognitive_operations": report.get("secondary_cognitive_operations", []),
        "recommended_synthesis_themes": synthesis_themes,
        "difficulty_distribution": difficulty_distribution,
    }, indent=2, ensure_ascii=False)

    return STAGE2_GENERATION_FROM_REPORT_PROMPT.format(
        format_description=fmt["description"],
        complexity_hint=fmt.get("complexity_hint", "Follow the dataset format specification."),
        json_skeleton=json.dumps(fmt["json_skeleton"], indent=2, ensure_ascii=False),
        report_summary=report_summary,
        primary_cognitive_operation=primary_op,
        common_error_patterns="\n".join(f"- {p}" for p in common_patterns) or "- failed to perform required cognitive operation",
        synthesis_themes="\n".join(f"- {t}" for t in synthesis_themes) or "- everyday social scenarios",
        difficulty=difficulty,
        format_specific_rules=format_specific_rules,
        target_count=target_count,
        difficulty_specific_rules=difficulty_specific_rules,
        distractor_strategy=distractor_strategy,
        retry_feedback=retry_feedback,
    )


