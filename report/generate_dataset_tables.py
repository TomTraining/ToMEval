from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from report.utils import collect_result_bundles, load_yaml
from report.utils.common import format_metric_value


BASE_METRICS = {"accuracy", "correct", "total"}


def _normalize_model_filters(models: Optional[List[Any]]) -> tuple[Optional[List[str]], Dict[str, str]]:
    if not models:
        return None, {}
    model_names: List[str] = []
    display_names: Dict[str, str] = {}
    for item in models:
        if isinstance(item, str):
            model_names.append(item)
            display_names[item] = item
        else:
            name = str(item["name"])
            display = str(item.get("display") or name)
            model_names.append(name)
            display_names[name] = display
    return model_names, display_names


def _basic_table(dataset: str, rows: Dict[str, Dict[str, Any]], ordered_models: List[str]) -> str:
    lines = [f"# {dataset} - 基础指标", ""]
    lines.append("| 指标 \\ 模型 | " + " | ".join(ordered_models) + " |")
    lines.append("|" + "|".join(["---"] * (len(ordered_models) + 1)) + "|")
    for metric in ("accuracy", "correct", "total"):
        values = [format_metric_value(rows.get(model, {}).get(metric, "-")) for model in ordered_models]
        lines.append("| " + " | ".join([metric, *values]) + " |")
    lines.append("")
    return "\n".join(lines)


def _other_table(dataset: str, rows: Dict[str, Dict[str, Any]], ordered_models: List[str]) -> str:
    scalar_keys = sorted(
        {
            key
            for metrics in rows.values()
            for key, value in metrics.items()
            if key not in BASE_METRICS and not isinstance(value, dict)
        }
    )
    dict_keys = sorted(
        {
            key
            for metrics in rows.values()
            for key, value in metrics.items()
            if isinstance(value, dict)
        }
    )

    lines = [f"# {dataset} - 其他指标", ""]

    if scalar_keys:
        lines.extend(["## 标量指标", ""])
        lines.append("| 指标 \\ 模型 | " + " | ".join(ordered_models) + " |")
        lines.append("|" + "|".join(["---"] * (len(ordered_models) + 1)) + "|")
        for key in scalar_keys:
            values = [format_metric_value(rows.get(model, {}).get(key, "-")) for model in ordered_models]
            lines.append("| " + " | ".join([key, *values]) + " |")
        lines.append("")

    for dict_key in dict_keys:
        lines.extend([f"## {dict_key}", ""])
        sub_keys = sorted(
            {
                sub_key
                for metrics in rows.values()
                for sub_key in (metrics.get(dict_key) or {}).keys()
            }
        )
        if not sub_keys:
            continue
        lines.append("| 子指标 \\ 模型 | " + " | ".join(ordered_models) + " |")
        lines.append("|" + "|".join(["---"] * (len(ordered_models) + 1)) + "|")
        for sub_key in sub_keys:
            values = []
            for model in ordered_models:
                metric_dict = rows.get(model, {}).get(dict_key, {})
                values.append(format_metric_value(metric_dict.get(sub_key, "-")))
            lines.append("| " + " | ".join([sub_key, *values]) + " |")
        lines.append("")

    if len(lines) == 2:
        lines.append("该数据集没有额外指标。")
        lines.append("")
    return "\n".join(lines)


def generate_dataset_tables(config_path: str) -> None:
    config = load_yaml(config_path)
    results_dir = str(config.get("results_dir", "results"))
    output_dir = Path(str(config.get("output_dir", "tables")))
    dataset_filter = config.get("dataset")
    if isinstance(dataset_filter, str):
        dataset_filter = [dataset_filter]

    model_filters, display_names = _normalize_model_filters(config.get("models"))
    bundles = collect_result_bundles(
        results_dir=results_dir,
        dataset_filter=dataset_filter,
        models_filter=model_filters,
        exp_suffix=config.get("exp_suffix"),
    )

    if not bundles:
        print("No metrics.json files found.")
        return

    for dataset, model_bundles in sorted(bundles.items()):
        dataset_dir = output_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)

        model_rows: Dict[str, Dict[str, Any]] = {}
        ordered_models: List[str] = []
        for model_name, bundle in sorted(model_bundles.items()):
            display_name = display_names.get(model_name, model_name)
            ordered_models.append(display_name)
            model_rows[display_name] = bundle["metrics"].get("avg_metrics", {})

            config_src = bundle["exp_dir"] / "config.json"
            if config_src.exists():
                model_dir = dataset_dir / display_name
                model_dir.mkdir(parents=True, exist_ok=True)
                config_payload = json.loads(config_src.read_text(encoding="utf-8"))
                config_payload["exp_dir"] = bundle["exp_dir"].name
                (model_dir / "config.json").write_text(
                    json.dumps(config_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

        (dataset_dir / "基础指标.md").write_text(
            _basic_table(dataset, model_rows, ordered_models),
            encoding="utf-8",
        )
        (dataset_dir / "其他指标.md").write_text(
            _other_table(dataset, model_rows, ordered_models),
            encoding="utf-8",
        )
        print(f"Saved tables for {dataset} -> {dataset_dir}")


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "tables_config.yaml")
    generate_dataset_tables(config_path)


if __name__ == "__main__":
    main()
