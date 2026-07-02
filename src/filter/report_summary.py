"""生成 filter 评估汇总报告。

读取所有数据集的 filter_summary.json 和 final/train_set.parquet，
生成易读的 Markdown 报告，展示每个阶段的过滤统计。
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _format_bad_reasons(label_counts_by_iter: List[Dict[str, Any]]) -> List[str]:
    """把各轮的 bad_reason_counts 渲染成逐轮明细行；无 bad 原因时返回空列表。

    兼容旧 summary（无 bad_reason_counts 字段时安全跳过）。
    """
    out: List[str] = []
    for iter_info in label_counts_by_iter:
        reasons = iter_info.get("bad_reason_counts") or {}
        if not reasons:
            continue
        detail = ", ".join(f"{k}={v}" for k, v in sorted(reasons.items()))
        out.append(f"- iter{iter_info.get('iter')}: {detail}\n")
    return out


def generate_summary_report(output_root: str, datasets: List[str]) -> str:
    """生成汇总报告（Markdown 格式）。

    Args:
        output_root: 评估输出根目录
        datasets: 数据集列表

    Returns:
        Markdown 格式的报告字符串
    """
    lines = ["# Data Eval 评估汇总报告\n"]

    for dataset in datasets:
        ds_root = Path(output_root) / dataset
        summary_path = ds_root / "filter_summary.json"

        if not summary_path.exists():
            lines.append(f"## {dataset}\n")
            lines.append("⚠️ 未找到评估结果\n")
            continue

        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        lines.append(f"## {dataset}\n")

        # 兼容新旧格式
        if "splits_run" in summary:
            # 新格式：按 split 处理
            lines.append(f"- **处理的 split 数**: {summary['splits_run']}/{summary['total_splits']}\n")

            # 统计每个 split 的情况
            for split_sum in summary.get("split_summaries", []):
                split_name = split_sum["split"]
                lines.append(f"\n### Split: `{split_name}`\n")
                lines.append(f"- **迭代轮数**: {split_sum['iters_run']}/{split_sum['max_iter']}\n")

                # 每轮的标签分布
                lines.append("\n**各轮标签分布**:\n")
                lines.append("| 轮次 | 总数 | easy | hard | medium | shortcut | bad | unfixable |\n")
                lines.append("|------|------|------|------|--------|----------|-----|----------|\n")

                for iter_info in split_sum.get("label_counts_by_iter", []):
                    iter_n = iter_info["iter"]
                    total = iter_info["total"]
                    labels = iter_info["label_counts"]

                    easy = labels.get("easy", 0)
                    hard = labels.get("hard", 0)
                    medium = labels.get("medium", 0)
                    shortcut = labels.get("shortcut", 0)
                    bad = labels.get("bad", 0)
                    unfixable = labels.get("unfixable", 0)

                    lines.append(
                        f"| iter{iter_n} | {total} | {easy} | {hard} | {medium} | "
                        f"{shortcut} | {bad} | {unfixable} |\n"
                    )

                # bad 细分原因明细（仅在存在 bad 时展示）
                bad_reason_lines = _format_bad_reasons(split_sum.get("label_counts_by_iter", []))
                if bad_reason_lines:
                    lines.append("\n**bad 细分原因**:\n")
                    lines.extend(bad_reason_lines)

                lines.append(f"\n- **保留样本数 (hard+medium)**: {split_sum['total_keep_pool']}\n")
        else:
            # 旧格式：整个数据集一起处理
            lines.append(f"- **迭代轮数**: {summary.get('iters_run', 0)}/{summary.get('max_iter', 0)}\n")

            lines.append("\n**各轮标签分布**:\n")
            lines.append("| 轮次 | 总数 | easy | hard | medium | shortcut | bad | unfixable |\n")
            lines.append("|------|------|------|------|--------|----------|-----|----------|\n")

            for iter_info in summary.get("label_counts_by_iter", []):
                iter_n = iter_info["iter"]
                total = iter_info["total"]
                labels = iter_info["label_counts"]

                easy = labels.get("easy", 0)
                hard = labels.get("hard", 0)
                medium = labels.get("medium", 0)
                shortcut = labels.get("shortcut", 0)
                bad = labels.get("bad", 0)
                unfixable = labels.get("unfixable", 0)

                lines.append(
                    f"| iter{iter_n} | {total} | {easy} | {hard} | {medium} | "
                    f"{shortcut} | {bad} | {unfixable} |\n"
                )

            lines.append(f"\n- **保留样本数 (hard+medium)**: {summary.get('total_keep_pool', 0)}\n")

        # 最终训练集统计
        train_set_path = ds_root / "final" / "train_set.parquet"
        if train_set_path.exists():
            train_df = pd.read_parquet(train_set_path)
            lines.append(f"\n### 最终训练集\n")
            lines.append(f"- **总样本数**: {len(train_df)}\n")
        else:
            lines.append(f"\n### 最终训练集\n")
            lines.append("⚠️ 未生成训练集\n")

        lines.append("\n---\n")

    # 全局汇总
    lines.append("\n## 全局汇总\n")
    lines.append("| 数据集 | 处理 splits | 保留样本 (hard+medium) | 最终训练集 |\n")
    lines.append("|--------|-------------|----------------------|------------|\n")

    for dataset in datasets:
        ds_root = Path(output_root) / dataset
        summary_path = ds_root / "filter_summary.json"
        train_set_path = ds_root / "final" / "train_set.parquet"

        if not summary_path.exists():
            lines.append(f"| {dataset} | - | - | - |\n")
            continue

        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        # 兼容新旧格式
        if "splits_run" in summary:
            splits_run = summary["splits_run"]
            total_splits = summary["total_splits"]
            total_keep = summary["total_keep_pool"]
            splits_info = f"{splits_run}/{total_splits}"
        else:
            splits_info = "1/1"
            total_keep = summary.get("total_keep_pool", 0)

        if train_set_path.exists():
            train_df = pd.read_parquet(train_set_path)
            final_count = len(train_df)
        else:
            final_count = 0

        lines.append(
            f"| {dataset} | {splits_info} | {total_keep} | {final_count} |\n"
        )

    return "".join(lines)


def save_summary_report(output_root: str, datasets: List[str], output_path: str) -> None:
    """生成并保存汇总报告。"""
    report = generate_summary_report(output_root, datasets)
    Path(output_path).write_text(report, encoding="utf-8")
    print(f"汇总报告已保存到: {output_path}")


if __name__ == "__main__":
    import yaml

    cfg = yaml.safe_load(Path("src/filter/config.yaml").read_text(encoding="utf-8"))
    datasets = cfg.get("datasets", [])
    output_root = cfg["paths"]["output_root"]

    report = generate_summary_report(output_root, datasets)
    print(report)

    # 保存到文件
    save_summary_report(output_root, datasets, f"{output_root}/SUMMARY_REPORT.md")
