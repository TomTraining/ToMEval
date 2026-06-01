"""生成 feedback 合成流水线汇总报告。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List


def generate_summary_report(
    stats: Dict[str, Dict],
    dataset_names: List[str],
    stage: str,
    model_name: str,
    output_path: Path,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Feedback 合成流水线汇总报告\n",
        f"- **时间**: {now}",
        f"- **Stage**: {stage}",
        f"- **合成模型**: {model_name}",
        f"- **数据集**: {', '.join(dataset_names)}",
        "",
        "## 各数据集统计\n",
        "| 数据集 | bad cases | 诊断报告 | 合成原始 | 合成通过 | 丢弃(测试集) | 丢弃(内部重复) |",
        "|--------|-----------|----------|----------|----------|-------------|----------------|",
    ]

    for ds in dataset_names:
        st = stats.get(ds, {})
        lines.append(
            f"| {ds} "
            f"| {st.get('bad_cases', '-')} "
            f"| {st.get('reports', '-')} "
            f"| {st.get('synthesized_raw', '-')} "
            f"| {st.get('synth_clean', '-')} "
            f"| {st.get('synth_dropped_test', '-')} "
            f"| {st.get('synth_dropped_internal', '-')} |"
        )

    # 合成样本文件统计
    lines.append("\n## 合成输出文件\n")
    synth_clean_root = output_path / "synth_clean"
    any_file = False
    for ds in dataset_names:
        ds_clean_dir = synth_clean_root / ds
        if not ds_clean_dir.exists():
            continue
        parquet_files = sorted(ds_clean_dir.glob("synthetic*.parquet"))
        if parquet_files:
            any_file = True
            lines.append(f"**{ds}**:")
            for pf in parquet_files:
                lines.append(f"  - `{pf}`")
    if not any_file:
        lines.append("*(暂无合成输出文件)*")

    return "\n".join(lines) + "\n"


def save_summary_report(
    stats: Dict[str, Dict],
    dataset_names: List[str],
    stage: str,
    model_name: str,
    output_path: Path,
) -> None:
    report = generate_summary_report(stats, dataset_names, stage, model_name, output_path)
    report_path = output_path / "SUMMARY.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"汇总报告已保存到: {report_path}")
