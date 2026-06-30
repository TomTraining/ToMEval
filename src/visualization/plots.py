"""通用评测可视化绘图。

设计原则：**数据集无关**——只消费 results/.../metrics.json（任意经
`hierarchical_metrics` 聚合的数据集都会产出嵌套的 `dimensions` 树），
以及其中的 per_sample_results。任何 ToMEval 数据集都能直接出图。

metrics 结构（见 src/evaluation/task_metrics.py）：
    avg_metrics.dimensions = {
        "<二级维度>": {"<split>": {"acc": float, "n": int,
                                   "dimensions": {...三级...}}},
        ...
    }
固定带一个 type 维度（按题型 prompt_type 切分）。

自动行为：
- 每个维度节点（含三级）-> 一张准确率柱状图（带样本数 n 标注）。
- 某维度各 split 都带同名单一子维度 -> 额外透视成 行×列 热力图
  （如 SocialMind dim1×dim2、EmoBench coarse×finegrained）。
- 顶层"分组均分"字典（键含 `_by_`，如 SocialMind q4_mean_score_by_dim）-> 均分柱状图。
- per_sample_results 中若含 judge1_score/judge2_score -> 出 judge 一致性图
  （散点 + Bland-Altman）；否则跳过。
- 多个 metrics.json -> 额外出多模型对比图。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import seaborn as sns  # noqa: E402
    _HAS_SNS = True
except Exception:  # pragma: no cover
    _HAS_SNS = False

# 尽量挑一个支持中文的字体（SocialMind 维度名等），挑不到就用默认，不报错。
for _font in ["PingFang SC", "Heiti SC", "Arial Unicode MS", "STHeiti", "SimHei"]:
    try:
        from matplotlib.font_manager import findfont, FontProperties
        if findfont(FontProperties(family=_font), fallback_to_default=False):
            plt.rcParams["font.sans-serif"] = [_font]
            break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False


# --------------------------------------------------------------------------- #
# 数据加载                                                                      #
# --------------------------------------------------------------------------- #
def load_metrics(path: str | Path) -> Dict[str, Any]:
    """接受 metrics.json 文件或包含它的目录，返回原始 payload。

    逐样本明细已从 metrics.json 拆到同目录的 detailed_metrics.jsonl（每行一条
    per_sample 结果）。这里把它读回来挂在 payload["per_sample_results"] 上，
    judge 一致性图等下游消费方无需感知拆分。
    """
    p = Path(path)
    if p.is_dir():
        p = p / "metrics.json"
    payload = json.loads(p.read_text(encoding="utf-8"))
    detailed_path = p.with_name("detailed_metrics.jsonl")
    if detailed_path.exists():
        with detailed_path.open(encoding="utf-8") as file:
            payload["per_sample_results"] = [json.loads(line) for line in file if line.strip()]
    return payload


def primary_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    """取 avg_metrics（metrics.json 现在只固化平均指标）。"""
    return payload.get("avg_metrics") or {}


def iter_dimension_nodes(
    metrics: Dict[str, Any],
) -> List[Tuple[str, Dict[str, Any]]]:
    """深度遍历 metrics["dimensions"] 树，返回 [(路径名, splits 字典), ...]。

    路径名形如 "ability" 或 "dim1/夫妻/dim2"（三级维度带父 split 前缀），用于图名与标题。
    splits 字典即 {split: {"acc", "n", ["dimensions"]}}。
    """
    out: List[Tuple[str, Dict[str, Any]]] = []

    def walk(dimensions: Dict[str, Any], prefix: str) -> None:
        if not isinstance(dimensions, dict):
            return
        for dim_name, splits in dimensions.items():
            if not isinstance(splits, dict) or not splits:
                continue
            path = f"{prefix}{dim_name}"
            out.append((path, splits))
            # 递归三级：各 split 下若挂了子 dimensions，带上 "父维度.split" 前缀继续。
            for split_key, split in splits.items():
                child = split.get("dimensions") if isinstance(split, dict) else None
                if isinstance(child, dict) and child:
                    walk(child, f"{path}/{split_key}/")

    walk(metrics.get("dimensions") or {}, "")
    return out


def split_rates_counts(splits: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, int]]:
    """从 splits 字典抽出准确率字典与样本数字典。"""
    rates = {k: float(v.get("acc", 0.0)) for k, v in splits.items() if isinstance(v, dict)}
    counts = {k: int(v.get("n", 0)) for k, v in splits.items() if isinstance(v, dict)}
    return rates, counts


def heatmap_from_splits(splits: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, float]]]:
    """若某维度每个 split 都恰好挂同一个子维度，则可透视成 行×列 热力图。

    返回 (子维度名, {"row|col": acc})；不满足条件返回 None。
    例：SocialMind dim1 各 split 下都有 dim2 -> dim1×dim2 热力图。
    """
    child_names = set()
    for split in splits.values():
        child = split.get("dimensions") if isinstance(split, dict) else None
        if not isinstance(child, dict) or len(child) != 1:
            return None
        child_names.add(next(iter(child)))
    if len(child_names) != 1:
        return None
    child_name = next(iter(child_names))
    rates: Dict[str, float] = {}
    for row_key, split in splits.items():
        sub_splits = split["dimensions"][child_name] or {}
        for col_key, sub in sub_splits.items():
            if isinstance(sub, dict):
                rates[f"{row_key}|{col_key}"] = float(sub.get("acc", 0.0))
    return (child_name, rates) if rates else None


def score_group_keys(metrics: Dict[str, Any]) -> List[str]:
    """非准确率的"分组均分"字典，如 SocialMind 的 q4_mean_score_by_dim(0-10 分)。

    约定：键里含 `_by_`（分组）但不以 `by_` 开头（那是历史准确率命名），值为非空 dict。"""
    return [
        k for k, v in metrics.items()
        if "_by_" in k and not k.startswith("by_") and not k.endswith("_counts")
        and isinstance(v, dict) and v
    ]


def all_per_sample(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """逐样本明细：由 load_metrics 从 detailed_metrics.jsonl 读到 payload 上。"""
    return payload.get("per_sample_results") or []


# --------------------------------------------------------------------------- #
# 单数据集绘图                                                                   #
# --------------------------------------------------------------------------- #
def plot_group_accuracy(
    group_name: str,
    rates: Dict[str, float],
    counts: Optional[Dict[str, int]],
    out_path: Path,
    title: Optional[str] = None,
) -> None:
    items = sorted(rates.items())
    labels = [str(k) for k, _ in items]
    values = [float(v) for _, v in items]
    width = max(6, min(0.5 * len(labels) + 2, 28))
    fig, ax = plt.subplots(figsize=(width, 5))
    bars = ax.bar(range(len(labels)), values, color="#4C8BF5")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title(title or group_name)
    for i, bar in enumerate(bars):
        n = (counts or {}).get(labels[i]) if counts else None
        tag = f"{values[i]:.2f}" + (f"\nn={n}" if n is not None else "")
        ax.text(bar.get_x() + bar.get_width() / 2, values[i] + 0.01, tag,
                ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_score_group(
    group_name: str,
    scores: Dict[str, float],
    out_path: Path,
    overall: Optional[float] = None,
) -> None:
    """分组平均分柱状图（非 0-1 准确率，y 轴按数据自适应）。"""
    items = sorted(scores.items())
    labels = [str(k) for k, _ in items]
    values = [float(v) for _, v in items]
    width = max(6, min(0.5 * len(labels) + 2, 28))
    fig, ax = plt.subplots(figsize=(width, 5))
    bars = ax.bar(range(len(labels)), values, color="#7E57C2")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, max(values) * 1.18 if values else 1)
    ax.set_ylabel("Mean score")
    title = group_name + (f"  (overall={overall:.2f})" if overall is not None else "")
    ax.set_title(title)
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2, values[i], f"{values[i]:.2f}",
                ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_heatmap(group_name: str, rates: Dict[str, float], out_path: Path) -> None:
    """组合键 "row|col" -> 热力图。"""
    rows: List[str] = sorted({str(k).split("|", 1)[0] for k in rates})
    cols: List[str] = sorted({str(k).split("|", 1)[1] for k in rates})
    matrix = [[rates.get(f"{r}|{c}") for c in cols] for r in rows]

    fig, ax = plt.subplots(figsize=(max(6, len(cols) * 1.2 + 2), max(4, len(rows) * 0.4 + 2)))
    if _HAS_SNS:
        sns.heatmap(
            [[(v if v is not None else float("nan")) for v in row] for row in matrix],
            xticklabels=cols, yticklabels=rows, annot=True, fmt=".2f",
            cmap="RdYlGn", vmin=0, vmax=1, cbar_kws={"label": "Accuracy"}, ax=ax,
        )
    else:
        data = [[(v if v is not None else float("nan")) for v in row] for row in matrix]
        im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right")
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
        for i in range(len(rows)):
            for j in range(len(cols)):
                if matrix[i][j] is not None:
                    ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, label="Accuracy")
    ax.set_title(group_name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_judge_agreement(per_sample: List[Dict[str, Any]], out_dir: Path) -> bool:
    """散点 + Bland-Altman。需要 per_sample 含 judge1_score/judge2_score。返回是否出图。"""
    pairs: List[Tuple[float, float]] = [
        (float(r["judge1_score"]), float(r["judge2_score"]))
        for r in per_sample
        if r.get("judge1_score") is not None and r.get("judge2_score") is not None
    ]
    if len(pairs) < 3:
        return False
    j1 = [a for a, _ in pairs]
    j2 = [b for _, b in pairs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # 散点 + y=x
    axes[0].scatter(j1, j2, alpha=0.5, color="#4C8BF5")
    lo, hi = min(j1 + j2), max(j1 + j2)
    axes[0].plot([lo, hi], [lo, hi], "r--", label="y=x")
    mae = sum(abs(a - b) for a, b in pairs) / len(pairs)
    axes[0].set_xlabel("Judge 1 score"); axes[0].set_ylabel("Judge 2 score")
    axes[0].set_title(f"Judge scatter (n={len(pairs)}, MAE={mae:.2f})")
    axes[0].legend()
    # Bland-Altman
    means = [(a + b) / 2 for a, b in pairs]
    diffs = [a - b for a, b in pairs]
    md = sum(diffs) / len(diffs)
    sd = (sum((d - md) ** 2 for d in diffs) / len(diffs)) ** 0.5
    axes[1].scatter(means, diffs, alpha=0.5, color="#F5934C")
    axes[1].axhline(md, color="k", label=f"mean={md:.2f}")
    axes[1].axhline(md + 1.96 * sd, color="gray", ls="--", label="+1.96SD")
    axes[1].axhline(md - 1.96 * sd, color="gray", ls="--", label="-1.96SD")
    axes[1].set_xlabel("Mean of judges"); axes[1].set_ylabel("Judge1 - Judge2")
    axes[1].set_title("Bland-Altman")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "judge_agreement.png", dpi=150)
    plt.close(fig)
    return True


def _safe_name(path_name: str) -> str:
    """把维度路径名（可能含 / 与空格）转成安全的文件名片段。"""
    return path_name.replace("/", "__").replace(" ", "_")


def render_single(payload: Dict[str, Any], out_dir: Path, prefix: str = "") -> List[Path]:
    """对单个 metrics payload 出全套图，返回生成的文件列表。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = primary_metrics(payload)
    written: List[Path] = []

    # 遍历 dimensions 树：每个维度的 splits 出一张准确率柱状图（带 n 标注）；
    # 若该维度各 split 恰好都挂同一个子维度，额外透视成 父×子 热力图。
    for path_name, splits in iter_dimension_nodes(metrics):
        rates, counts = split_rates_counts(splits)
        if not rates:
            continue
        name = _safe_name(path_name)
        bar_path = out_dir / f"{prefix}{name}.png"
        plot_group_accuracy(path_name, rates, counts, bar_path)
        written.append(bar_path)

        heatmap = heatmap_from_splits(splits)
        if heatmap is not None:
            child_name, hm_rates = heatmap
            hm_path = out_dir / f"{prefix}{name}_x_{_safe_name(child_name)}_heatmap.png"
            plot_heatmap(f"{path_name} × {child_name}", hm_rates, hm_path)
            written.append(hm_path)

    # 分组平均分（如 SocialMind Q4 rubric：q4_mean_score_by_dim，0-10 分）。
    for sk in score_group_keys(metrics):
        scores = metrics[sk]
        overall_key = sk.split("_by_", 1)[0]  # q4_mean_score_by_dim -> q4_mean_score
        overall = metrics.get(overall_key)
        overall = float(overall) if isinstance(overall, (int, float)) else None
        path = out_dir / f"{prefix}{sk}.png"
        plot_score_group(sk, scores, path, overall)
        written.append(path)

    if plot_judge_agreement(all_per_sample(payload), out_dir):
        written.append(out_dir / "judge_agreement.png")
    return written


# --------------------------------------------------------------------------- #
# 多模型对比                                                                     #
# --------------------------------------------------------------------------- #
def plot_radar_comparison(
    group_name: str,
    keys: List[str],
    labels: List[str],
    rates_list: List[Dict[str, float]],
    out_path: Path,
) -> None:
    """多模型在某维度(类别>=3)上的准确率雷达图。rates_list[i] 为第 i 个模型的 {split: acc}。"""
    n = len(keys)
    angles = [i / n * 2 * math.pi for i in range(n)]
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(max(6, n * 0.5 + 3), max(6, n * 0.5 + 3)),
                           subplot_kw={"polar": True})
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(keys, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_rlabel_position(180 / n)
    for lab, rates in zip(labels, rates_list):
        vals = [float(rates.get(k, 0.0)) for k in keys]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=1.5, label=lab)
        ax.fill(angles, vals, alpha=0.12)
    ax.set_title(f"{group_name} by model", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_model_comparison(
    labeled: List[Tuple[str, Dict[str, Any]]],
    out_dir: Path,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    labels = [lab for lab, _ in labeled]
    metrics_list = [primary_metrics(p) for _, p in labeled]

    # 总体准确率对比
    overall = [float(m.get("accuracy", 0.0)) for m in metrics_list]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2 + 2), 5))
    bars = ax.bar(range(len(labels)), overall, color="#4C8BF5")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy"); ax.set_title("Overall accuracy by model")
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2, overall[i] + 0.01, f"{overall[i]:.3f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    p = out_dir / "model_comparison_overall.png"
    fig.savefig(p, dpi=150); plt.close(fig); written.append(p)

    # 各共享维度（按维度路径名对齐）：分组柱状图（模型并排）。
    # 每个模型的 dimensions 树展平成 {路径名: {split: acc}}，取各模型都有的路径。
    per_model_dims: List[Dict[str, Dict[str, float]]] = []
    for m in metrics_list:
        flat: Dict[str, Dict[str, float]] = {}
        for path, splits in iter_dimension_nodes(m):
            rates, _ = split_rates_counts(splits)
            flat[path] = rates
        per_model_dims.append(flat)

    shared = set(per_model_dims[0])
    for flat in per_model_dims[1:]:
        shared &= set(flat)
    for path in sorted(shared):
        keys = sorted(set().union(*[set(flat.get(path, {}).keys()) for flat in per_model_dims]))
        if not keys:
            continue
        safe = path.replace("/", "_")
        fig, ax = plt.subplots(figsize=(max(7, len(keys) * 0.6 * len(labels) + 2), 5))
        n = len(labels)
        bw = 0.8 / n
        for mi, (lab, flat) in enumerate(zip(labels, per_model_dims)):
            vals = [float(flat.get(path, {}).get(k, 0.0)) for k in keys]
            xs = [j + (mi - (n - 1) / 2) * bw for j in range(len(keys))]
            ax.bar(xs, vals, width=bw, label=lab)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy"); ax.set_title(f"{path} by model")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = out_dir / f"model_comparison_{safe}.png"
        fig.savefig(p, dpi=150); plt.close(fig); written.append(p)

        # 类别数 >=3 时额外出雷达图（如二级维度 dim2 的多模型对比）。
        if len(keys) >= 3:
            rp = out_dir / f"model_comparison_{safe}_radar.png"
            plot_radar_comparison(path, keys, labels, [flat.get(path, {}) for flat in per_model_dims], rp)
            written.append(rp)
    return written
