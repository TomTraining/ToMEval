"""通用评测可视化模块。

用法：
    python -m src.visualization --results results/SocialToM/<model>/<exp> --out figures/SocialToM
    python -m src.visualization --results results/A/... results/B/... --out figures/cmp

只读 metrics.json，数据集无关。详见 plots.py。
"""

from .plots import (
    load_metrics,
    primary_metrics,
    render_single,
    plot_model_comparison,
)

__all__ = [
    "load_metrics",
    "primary_metrics",
    "render_single",
    "plot_model_comparison",
]
