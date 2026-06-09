from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.llm.client import LLMResponse

from .lang import get_sample_lang
from .prompts import build_option_bundle, build_prompt, prompt_type
from .protocols import extractor_name_for, shuffle_for, system_prompt_for
from .storage import serialize_llm_response
from .types import PredictionRecord, StandardizedSample


def predict_records(
    samples: List[StandardizedSample],
    dataset_name: str,
    client: Any,
    repeats: int,
    protocol: Optional[str] = None,
) -> List[PredictionRecord]:
    records: List[PredictionRecord] = []
    # 协议评测下：选项是否 shuffle、答题指令是否进 user prompt、用哪个 extractor 全部由协议决定。
    use_shuffle = shuffle_for(protocol)
    extractor = extractor_name_for(protocol)

    for repeat in range(repeats):
        repeat_payloads: List[PredictionRecord] = []
        # 按题型分批调用模型，避免开放题、单选题、多选题混在一起影响输出稳定性。
        grouped_prompts: Dict[str, List[str]] = {"open": [], "mcq_single": [], "mcq_multi": []}
        grouped_system: Dict[str, List[str]] = {"open": [], "mcq_single": [], "mcq_multi": []}
        grouped_indices: Dict[str, List[int]] = {"open": [], "mcq_single": [], "mcq_multi": []}

        for sample_index, sample in enumerate(samples):
            option_map, correct_letters, wrong_letters, shuffle_seed = build_option_bundle(
                dataset_name,
                sample["sample_id"],
                sample["answer"],
                repeat,
                shuffle=use_shuffle,
            )
            current_prompt_type = prompt_type(sample["answer"])
            # 设了协议：答题格式指令改由 system prompt 承载，user prompt 去掉 answer_instruction。
            prompt = build_prompt(sample, option_map, include_instruction=protocol is None)
            if protocol is None:
                system_prompt = ""
            else:
                lang = get_sample_lang(sample.get("meta"))
                system_prompt = system_prompt_for(protocol, lang, current_prompt_type)

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
                "protocol": protocol,
                "extractor": extractor,
            }
            repeat_payloads.append(payload)
            grouped_prompts[current_prompt_type].append(prompt)
            grouped_system[current_prompt_type].append(system_prompt)
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
            # 未设协议时 system_prompts=None，沿用客户端单一 system_prompt（旧行为）。
            system_prompts = grouped_system[current_prompt_type] if protocol is not None else None
            responses = client.batch_generate(prompts, desc=desc, system_prompts=system_prompts)
            for sample_index, response in zip(grouped_indices[current_prompt_type], responses):
                responses_by_index[sample_index] = response

        for payload in repeat_payloads:
            # prediction.jsonl 只保留原始模型输出，答案提取统一放在 judge 阶段进行。
            response = responses_by_index.get(payload["sample_index"])
            payload["pred"] = serialize_llm_response(response)
            records.append(payload)

    return records
