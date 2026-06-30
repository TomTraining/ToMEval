from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.llm.client import LLMResponse

from .paths import average_metrics


def pred_content(record: Dict[str, Any]) -> Any:
    """取记录里模型的原始输出文本(pred.content);pred 缺失或为 None 时返回 None。

    判分各处都从 record["pred"]["content"] 取模型输出，统一收口到这里，
    避免 `(record.get("pred") or {}).get("content")` 在多个模块里重复。
    """
    return (record.get("pred") or {}).get("content")


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
    """落盘评测指标，拆成两份：

    - metrics.json        只放跨 repeat 平均后的 avg_metrics（含 dimensions 树）。
    - detailed_metrics.jsonl  逐样本明细：把各 repeat 的 per_sample_results 平铺，每行一条
      （每条自带 sample_id / repeat / prompt_type，足以还原归属）。

    per_sample_results 不再进 metrics.json；all_metrics 里的 accuracy/correct/total 等标量
    已被 avg_metrics 覆盖，故也不重复落盘。
    """
    # avg_metrics 只平均数值与嵌套 dict，per_sample_results（list）天然不会进去，无需额外清理。
    metrics_payload = {"avg_metrics": average_metrics(all_metrics)}
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    detailed_path = output_dir / "detailed_metrics.jsonl"
    with detailed_path.open("w", encoding="utf-8") as file:
        for metrics in all_metrics:
            for item in metrics.get("per_sample_results") or []:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")
    return metrics_path
