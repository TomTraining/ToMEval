"""Minimal shared helpers for the standardized QA evaluation flow."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import yaml

from src.dataloader import load_dataset


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    return {
        "llm_config": config.get("llm", {}),
        "repeats": config.get("repeats", 1),
        "max_samples": config.get("max_samples", 0),
        "normalized_datasets_path": config.get("normalized_datasets_path", "datasets_normalized"),
        "results_path": config.get("results_path", "results"),
        "judge_config": config.get("judge", {}),
    }


def create_llm_client(llm_config: Dict[str, Any], dataset_config: Optional[Dict[str, Any]] = None) -> Any:
    config = llm_config.copy()
    if dataset_config and dataset_config.get("system_prompt"):
        config["system_prompt"] = dataset_config["system_prompt"]
    from src.llm import StructureClient

    return StructureClient.from_config(config)


def create_content_client(llm_config: Dict[str, Any], dataset_config: Optional[Dict[str, Any]] = None) -> Any:
    config = llm_config.copy()
    if dataset_config and dataset_config.get("system_prompt"):
        config["system_prompt"] = dataset_config["system_prompt"]
    from src.llm import ContentClient

    return ContentClient.from_config(config)


def load_and_limit_data(
    subset: str,
    datasets_root: str = "datasets_normalized",
    max_samples: int = 0,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    import random

    data = load_dataset(subset, datasets_root=datasets_root)
    if max_samples > 0:
        random.seed(seed)
        data = random.sample(data, min(max_samples, len(data)))
    return data
