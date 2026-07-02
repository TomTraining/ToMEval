"""
Summary Generator - HTML Version

从 tables/ 目录生成支持合并单元格的 HTML 总览表格 SUMMARY.md。

读 tables/、解析基础指标、归并 model/judge 的共享逻辑见 report/utils/summary_tables.py，
本文件只负责把归并结果渲染成带合并单元格的 HTML 表格。
"""

import sys
from html import escape
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.report.utils import (
    collect_metrics_from_tables,
    format_accuracy,
    split_model_judge,
)


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
