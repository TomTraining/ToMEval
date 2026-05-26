"""
Summary Generator - HTML Version

从 tables/ 目录生成支持合并单元格的 HTML 总览表格 SUMMARY.md。
"""

import json
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def config_display_name(config_path: Path, dataset_dir: Path) -> str:
    config_data = read_json(config_path)
    table_display = str(config_data.get("table_display") or "").strip()
    if table_display:
        return table_display

    rel_parent = config_path.parent.relative_to(dataset_dir)
    parts = rel_parent.parts
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return parts[0] if parts else config_path.parent.name


def collect_config_display_names(dataset_dir: Path) -> List[str]:
    labels: List[str] = []

    for config_path in sorted(dataset_dir.glob("*/*/config.json")):
        label = config_display_name(config_path, dataset_dir)
        if label not in labels:
            labels.append(label)

    for config_path in sorted(dataset_dir.glob("*/config.json")):
        label = config_display_name(config_path, dataset_dir)
        if label not in labels:
            labels.append(label)

    return labels


def resolve_metric_label(raw_label: str, config_labels: List[str]) -> Optional[str]:
    if raw_label in config_labels:
        return raw_label

    matches = [label for label in config_labels if label == raw_label or label.startswith(f"{raw_label}/")]
    if len(matches) == 1:
        return matches[0]

    return raw_label


def parse_basic_metrics_table(table_dir: Path) -> Dict[str, Dict[str, Any]]:
    basic_metrics_file = table_dir / "基础指标.md"
    if not basic_metrics_file.exists():
        return {}

    metrics: Dict[str, Dict[str, Any]] = {}
    config_labels = collect_config_display_names(table_dir)
    content = basic_metrics_file.read_text(encoding="utf-8")

    lines = content.strip().split("\n")
    data_lines = [line for line in lines if line.startswith("|") and "---" not in line]

    if len(data_lines) < 2:
        return {}

    header = [cell.strip() for cell in data_lines[0].split("|")[1:-1]]
    models = [resolve_metric_label(model, config_labels) for model in header[1:]]

    for line in data_lines[1:]:
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if len(cells) != len(models) + 1:
            continue

        metric_name = cells[0]
        for i, model in enumerate(models):
            if not model:
                continue

            value: Any = cells[i + 1]
            if value == "-":
                continue

            try:
                value = float(value) if "." in value else int(value)
            except ValueError:
                pass

            metrics.setdefault(model, {})[metric_name] = value

    return metrics


def collect_metrics_from_tables(tables_dir: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    tables_path = Path(tables_dir)
    metrics_data: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for dataset_dir in tables_path.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name.startswith("."):
            continue

        dataset_metrics = parse_basic_metrics_table(dataset_dir)
        if dataset_metrics:
            metrics_data[dataset_dir.name] = dataset_metrics

    return metrics_data


def split_model_judge(label: str) -> Tuple[str, str]:
    if "/" not in label:
        return label.strip(), "unknown"

    model_label, judge_label = label.split("/", 1)
    judge_label = judge_label.strip()
    lowered = judge_label.lower()

    for prefix in ("judge-", "judge:", "judge_"):
        if lowered.startswith(prefix):
            judge_label = judge_label[len(prefix):].strip()
            break

    return model_label.strip(), judge_label or "unknown"


def format_accuracy(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value in (None, ""):
        return "-"
    return str(value)


def generate_summary_table(tables_dir: str) -> str:
    metrics_data = collect_metrics_from_tables(tables_dir)

    datasets = sorted(metrics_data.keys())
    generate_models = set()
    judges = set()
    grouped: Dict[str, Dict[str, Dict[str, str]]] = {}

    for dataset, dataset_metrics in metrics_data.items():
        for combined_label, metric_values in dataset_metrics.items():
            generate_model, judge_model = split_model_judge(combined_label)
            generate_models.add(generate_model)
            judges.add(judge_model)
            grouped.setdefault(judge_model, {}).setdefault(dataset, {})[generate_model] = format_accuracy(
                metric_values.get("accuracy", "-")
            )

    models = sorted(generate_models)
    judge_models = sorted(judges)

    if not datasets or not models or not judge_models:
        return "## 总览表格\n\n没有找到任何基础指标文件。\n"

    lines = [
        "## 总览表格：Accuracy",
        "",
        "<table>",
        "  <thead>",
        "    <tr>",
        "      <th>数据集 \\ 模型</th>",
    ]

    for model in models:
        lines.append(f"      <th>{escape(model)}</th>")

    lines.extend([
        "    </tr>",
        "  </thead>",
        "  <tbody>",
    ])

    colspan = len(models) + 1

    for judge_model in judge_models:
        lines.append(
            f'    <tr><td colspan="{colspan}" align="center"><strong>Judge: {escape(judge_model)}</strong></td></tr>'
        )

        for dataset in datasets:
            lines.append("    <tr>")
            lines.append(f"      <td>{escape(dataset)}</td>")

            for model in models:
                value = grouped.get(judge_model, {}).get(dataset, {}).get(model, "-")
                lines.append(f'      <td align="right">{escape(value)}</td>')

            lines.append("    </tr>")

    lines.extend([
        "  </tbody>",
        "</table>",
        "",
    ])

    return "\n".join(lines)


def generate_summary(tables_dir: str = "tables", output_file: str = None) -> str:
    summary = generate_summary_table(tables_dir)

    if output_file is not None:
        output_path = Path(output_file)
        output_path.write_text(summary, encoding="utf-8")
        print(f"总览表格已保存到: {output_path}")

    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="从 tables/ 目录生成 HTML 总览汇总表格")
    parser.add_argument("--tables-dir", type=str, default="tables")
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args()

    output_file = args.output_file
    if output_file is None:
        output_file = Path(args.tables_dir) / "SUMMARY-html.md"

    generate_summary(args.tables_dir, output_file)

    if args.stdout:
        print(generate_summary_table(args.tables_dir))


if __name__ == "__main__":
    main()