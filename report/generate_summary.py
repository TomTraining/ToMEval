from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from report.utils import parse_markdown_table


def parse_basic_metrics_table(dataset_dir: Path) -> Dict[str, Dict[str, str]]:
    basic_metrics_path = dataset_dir / "基础指标.md"
    if not basic_metrics_path.exists():
        return {}
    parsed = parse_markdown_table(basic_metrics_path.read_text(encoding="utf-8"))
    model_metrics: Dict[str, Dict[str, str]] = {}
    for metric_name, row in parsed.items():
        for model_name, value in row.items():
            model_metrics.setdefault(model_name, {})[metric_name] = value
    return model_metrics


def generate_summary_markdown(tables_dir: str) -> str:
    tables_path = Path(tables_dir)
    dataset_metrics: Dict[str, Dict[str, Dict[str, str]]] = {}
    all_models = set()

    for dataset_dir in sorted(path for path in tables_path.iterdir() if path.is_dir()):
        parsed = parse_basic_metrics_table(dataset_dir)
        if not parsed:
            continue
        dataset_metrics[dataset_dir.name] = parsed
        all_models.update(parsed.keys())

    models = sorted(all_models)
    if not dataset_metrics or not models:
        return "## 总览表格\n\n没有可用的基础指标表。\n"

    lines = [
        "## 总览表格：Accuracy",
        "",
        "| 数据集 \\ 模型 | " + " | ".join(models) + " |",
        "|" + "|".join(["---"] + ["-:"] * len(models)) + "|",
    ]
    for dataset in sorted(dataset_metrics):
        row = [dataset]
        for model in models:
            row.append(dataset_metrics[dataset].get(model, {}).get("accuracy", "-"))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-dir", default="tables")
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_file) if args.output_file else Path(args.tables_dir) / "SUMMARY.md"
    markdown = generate_summary_markdown(args.tables_dir)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Saved summary to: {output_path}")
    if args.stdout:
        print(markdown)


if __name__ == "__main__":
    main()
