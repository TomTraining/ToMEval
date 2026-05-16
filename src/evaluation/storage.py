from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.llm.client import LLMResponse

from .paths import average_metrics


def serialize_llm_response(response: LLMResponse | None) -> Dict[str, Any]:
    if response is None:
        return {"content": None, "reasoning": ""}

    content = response.content
    if content is None:
        serialized_content = None
    elif hasattr(content, "model_dump"):
        serialized_content = content.model_dump()
    else:
        serialized_content = content

    return {
        "content": serialized_content,
        "reasoning": response.reasoning,
    }


def write_prediction_file(output_dir: Path, records: List[Dict[str, Any]]) -> Path:
    prediction_path = output_dir / "prediction.jsonl"
    with prediction_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return prediction_path


def read_prediction_file(prediction_path: Path) -> List[Dict[str, Any]]:
    with prediction_path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def save_metrics(output_dir: Path, all_metrics: List[Dict[str, Any]]) -> Path:
    metrics_path = output_dir / "metrics.json"
    # table 阶段只读取 metrics.json，所以这里统一固化 repeat 明细和平均指标。
    metrics_payload = {
        "avg_metrics": average_metrics(all_metrics),
        "all_metrics": all_metrics,
    }
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics_path
