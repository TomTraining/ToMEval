from __future__ import annotations

from typing import Any, Dict, List

from src.llm.client import LLMResponse

from .prompts import build_option_bundle, build_prompt, extract_prediction_value, prompt_type
from .storage import serialize_llm_response
from .types import PredictionRecord, StandardizedSample


def predict_records(
    samples: List[StandardizedSample],
    dataset_name: str,
    client: Any,
    repeats: int,
) -> List[PredictionRecord]:
    records: List[PredictionRecord] = []

    for repeat in range(repeats):
        repeat_payloads: List[PredictionRecord] = []
        # 按题型分批调用模型，避免开放题、单选题、多选题混在一起影响输出稳定性。
        grouped_prompts: Dict[str, List[str]] = {"open": [], "mcq_single": [], "mcq_multi": []}
        grouped_indices: Dict[str, List[int]] = {"open": [], "mcq_single": [], "mcq_multi": []}

        for sample_index, sample in enumerate(samples):
            option_map, correct_letters, wrong_letters, shuffle_seed = build_option_bundle(
                dataset_name,
                sample["sample_id"],
                sample["answer"],
                repeat,
            )
            current_prompt_type = prompt_type(sample["answer"])
            prompt = build_prompt(sample, option_map)

            payload: PredictionRecord = {
                "sample_id": sample["sample_id"],
                "sample_index": sample_index,
                "repeat": repeat,
                "story": sample["story"],
                "question": sample["question"],
                "correct_answers": sample["answer"]["correct_answers"],
                "wrong_answers": sample["answer"]["wrong_answers"],
                "prompt_type": current_prompt_type,
                "prompt": prompt,
                "options": option_map,
                "correct_letters": correct_letters,
                "wrong_letters": wrong_letters,
                "shuffle_seed": shuffle_seed,
                "meta": sample["meta"],
                "pred": {"content": None, "reasoning": ""},
                "raw_prediction": None,
            }
            repeat_payloads.append(payload)
            grouped_prompts[current_prompt_type].append(prompt)
            grouped_indices[current_prompt_type].append(sample_index)

        responses_by_index: Dict[int, LLMResponse] = {}
        prompt_type_labels = {
            "open": "Open QA",
            "mcq_single": "Multiple Choice (Single)",
            "mcq_multi": "Multiple Choice (Multi)"
        }
        for current_prompt_type in ("open", "mcq_single", "mcq_multi"):
            prompts = grouped_prompts[current_prompt_type]
            if not prompts:
                continue
            desc = f"Generating ({prompt_type_labels[current_prompt_type]})"
            responses = client.batch_generate(prompts, desc=desc)
            for sample_index, response in zip(grouped_indices[current_prompt_type], responses):
                responses_by_index[sample_index] = response

        for payload in repeat_payloads:
            # prediction.jsonl 同时保留原始模型输出和规整后的 raw_prediction，方便复查。
            response = responses_by_index.get(payload["sample_index"])
            payload["pred"] = serialize_llm_response(response)
            payload["raw_prediction"] = extract_prediction_value(payload["prompt_type"], response)
            records.append(payload)

    return records
