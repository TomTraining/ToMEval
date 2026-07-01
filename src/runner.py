"""Minimal shared helpers for the standardized QA evaluation flow."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import yaml

from src.dataloader import load_dataset

logger = logging.getLogger(__name__)


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    return value


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, encoding="utf-8") as file:
        config = _expand_env_vars(yaml.safe_load(file) or {})

    llm_config = dict(config.get("llm", {}))
    repeats = config.get("repeats", 1)
    protocol = config.get("protocol")

    # 设了协议:采样参数(temperature/top_p/max_tokens/enable_thinking)由协议权威表覆盖，
    # repeats 取协议的 n_samples(direct/direct_think/cot→1, del_tom→8)，忽略 config 的 repeats。
    if protocol is not None:
        from src.evaluation.protocols import llm_overrides_for, repeats_for

        llm_config.update(llm_overrides_for(protocol))
        derived_repeats = repeats_for(protocol, repeats)
        if derived_repeats != repeats:
            logger.info(
                f"Protocol '{protocol}': repeats overridden {repeats} -> {derived_repeats} (= n_samples)."
            )
        repeats = derived_repeats
        logger.info(f"Protocol '{protocol}': llm sampling overridden to {llm_overrides_for(protocol)}.")

    return {
        "llm_config": llm_config,
        "repeats": repeats,
        "max_samples": config.get("max_samples", 0),
        "normalized_datasets_path": config.get("normalized_datasets_path", "datasets_normalized"),
        "results_path": config.get("results_path", "results"),
        "figures_path": config.get("figures_path", "figures"),
        # judge 参数不再由 experiment_config 管理，移到各数据集 tasks/<DS>/config.yaml 的 judge1/judge2 段。
        "protocol": protocol,
        # stage 与 datasets 由配置驱动:stage 控制 predict/metric/all,datasets 供 run_eval 批量评测。
        "stage": config.get("stage", "all"),
        "datasets": list(config.get("datasets") or []),
    }


def create_llm_client(llm_config: Dict[str, Any]) -> Any:
    config = _expand_env_vars(llm_config.copy())
    from src.llm import StructureClient

    return StructureClient.from_config(config)


def create_content_client(llm_config: Dict[str, Any]) -> Any:
    config = _expand_env_vars(llm_config.copy())
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
