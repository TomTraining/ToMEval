from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def find_experiment_dir(results_dir: str, dataset: str, model_name: str, exp_suffix: str | None = None) -> Optional[Path]:
    base_dir = Path(results_dir) / dataset / model_name
    if not base_dir.exists():
        return None
    if exp_suffix:
        target = base_dir / f"exp_{exp_suffix}"
        return target if target.exists() else None
    candidates = sorted(base_dir.glob("exp_*"))
    return candidates[-1] if candidates else None


def load_metrics_payload(exp_dir: Path) -> Dict[str, Any]:
    metrics_path = exp_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found: {metrics_path}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def load_prediction_records(exp_dir: Path) -> List[Dict[str, Any]]:
    prediction_path = exp_dir / "prediction.jsonl"
    if not prediction_path.exists():
        raise FileNotFoundError(f"prediction.jsonl not found: {prediction_path}")
    with prediction_path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def iter_datasets(results_dir: str) -> Iterable[str]:
    base_dir = Path(results_dir)
    if not base_dir.exists():
        return []
    return [path.name for path in sorted(base_dir.iterdir()) if path.is_dir()]


def collect_result_bundles(
    results_dir: str,
    dataset_filter: Optional[List[str]] = None,
    models_filter: Optional[List[str]] = None,
    exp_suffix: str | None = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    bundles: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for dataset in iter_datasets(results_dir):
        if dataset_filter and dataset not in dataset_filter:
            continue
        dataset_dir = Path(results_dir) / dataset
        model_bundles: Dict[str, Dict[str, Any]] = {}
        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            if models_filter and model_name not in models_filter:
                continue
            exp_dir = find_experiment_dir(results_dir, dataset, model_name, exp_suffix)
            if exp_dir is None:
                continue
            metrics_path = exp_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            model_bundles[model_name] = {
                "exp_dir": exp_dir,
                "metrics": load_metrics_payload(exp_dir),
            }
        if model_bundles:
            bundles[dataset] = model_bundles
    return bundles
