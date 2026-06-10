"""CLI：python -m src.visualization --results <path...> --out <dir>

--results 接受一个或多个 results 实验目录（或直接是 metrics.json）。
- 单个：出该模型全套分组图（柱状图 + 组合键热力图 + judge 一致性图，按需自动）。
- 多个：额外出多模型对比图。标签默认取路径中 results/<dataset>/<model>/... 的 <model> 段。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

from .plots import load_metrics, render_single, plot_model_comparison


def _label_for(path: str) -> str:
    parts = Path(path).resolve().parts
    if "results" in parts:
        i = parts.index("results")
        # results/<dataset>/<model>/<exp>
        if len(parts) > i + 2:
            return parts[i + 2]
    return Path(path).name


def main() -> None:
    parser = argparse.ArgumentParser(description="ToMEval 通用评测可视化")
    parser.add_argument("--results", nargs="+", required=True,
                        help="一个或多个 results 实验目录或 metrics.json 路径")
    parser.add_argument("--out", default="figures", help="图片输出目录")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    labeled: list[Tuple[str, dict]] = []
    for res in args.results:
        payload = load_metrics(res)
        label = _label_for(res)
        labeled.append((label, payload))
        sub = out_dir / label if len(args.results) > 1 else out_dir
        written = render_single(payload, sub, prefix="")
        print(f"[{label}] 生成 {len(written)} 张图 -> {sub}")
        for p in written:
            print(f"    {p}")

    if len(labeled) > 1:
        cmp_files = plot_model_comparison(labeled, out_dir / "comparison")
        print(f"[comparison] 生成 {len(cmp_files)} 张对比图 -> {out_dir / 'comparison'}")
        for p in cmp_files:
            print(f"    {p}")


if __name__ == "__main__":
    main()
