"""V3 数据飞轮决策树编排器。

以 split（单个 parquet 文件）为最小处理单元，每个 split 独立迭代评估+修复。

    [输入] train_datasets/{dataset}/{split}.parquet
        ↓ Phase B 全量 pass@k
        ├─ all_passed                    → label=easy, repair_type=easy
        ├─ partial / all_failed
        │     ↓ Phase C answerability
        │     ├─ unanswerable             → label=bad, repair_type=unanswerable
        │     ├─ all_failed + answerable  → label=hard, repair_type=None
        │     └─ partial + answerable
        │           ↓ Phase D shortcut 三探测
        │           ├─ is_shortcut         → label=shortcut, repair_type=shortcut
        │           └─ not_shortcut        → label=medium, repair_type=None
        ↓ Phase E 修复（label∈{easy,bad,shortcut}）→ 下一轮输入
        ↓ 最后一轮修复产出 → 额外纯评估轮（不修复）

    [输出] filter_output/{dataset}/split_{stem}/
              eval_iter{N}/passk.parquet
              eval_iter{N}/answerability.parquet
              eval_iter{N}/shortcut.parquet
              eval_iter{N}/labels.parquet
              eval_iter{N}/repaired.parquet
              eval_iter{N}/summary.json
              sampled_input.parquet   ← iter=1 采样后保存，finalize 用
           filter_output/{dataset}/final/train_set.parquet
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from filter.base import (
    _CONFIG_PATH,
    load_answer_models,
    load_judge_client,
)
from filter.eval import (
    _BUCKET_ALL_FAILED,
    _BUCKET_ALL_PASSED,
    _BUCKET_PARTIAL,
    run_answerability_on_df,
    run_passk_on_df,
    run_shortcut_on_df,
    write_answerability_parquet,
    write_passk_parquet,
    write_shortcut_parquet,
)
from filter.repair import (
    repair_samples,
    write_repaired_parquet,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# V3 数据飞轮固定参数（不需要用户配置）
# ──────────────────────────────────────────────────────────────────────────────

_PASS_K = 3                    # pass@k 的 k 值
_MAX_ITER_DEFAULT = 3          # 默认修复迭代上限
_SHORTCUT_ENABLED = True       # 是否启用 shortcut 检测
_SHORTCUT_THRESHOLD = "majority"  # majority = ceil(k/2)
_SHORTCUT_DIMENSIONS = ["no_story", "no_question", "no_options"]
_SKIP_ANSWERABILITY_ON_ALL_PASSED = True  # all_passed 直接判 easy
_REPAIR_ENABLED_TYPES = ["unanswerable", "easy", "shortcut"]

_LABEL_EASY = "easy"
_LABEL_HARD = "hard"
_LABEL_MEDIUM = "medium"
_LABEL_SHORTCUT = "shortcut"
_LABEL_BAD = "bad"
_LABEL_UNFIXABLE = "unfixable"
_KEEP_LABELS = {_LABEL_HARD, _LABEL_MEDIUM}


# ──────────────────────────────────────────────────────────────────────────────
# 配置 & 输入加载
# ──────────────────────────────────────────────────────────────────────────────


def load_filter_config() -> Dict[str, Any]:
    """读 filter/config.yaml，提取 paths、sampling 和 max_iter 配置。"""
    cfg = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))

    paths = cfg.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("config.yaml 缺 paths 段")

    required = ("input_root", "output_root")
    missing = [k for k in required if k not in paths]
    if missing:
        raise ValueError(f"config.yaml paths 缺字段: {missing}")

    sampling = cfg.get("sampling") or {}
    max_iter = cfg.get("max_iter", _MAX_ITER_DEFAULT)

    return {
        "input_root": str(paths["input_root"]),
        "output_root": str(paths["output_root"]),
        "sampling_enabled": bool(sampling.get("enabled", False)),
        "max_samples": int(sampling.get("max_samples", 100)),
        "max_iter": int(max_iter),
    }


def _split_dir(output_root: str, dataset: str, split_stem: str) -> Path:
    """返回某个 split 的输出根目录（中间产物）。"""
    return Path(output_root) / "_intermediate" / dataset / f"split_{split_stem}"


def _iter_dir(split_root: Path, iter_n: int) -> Path:
    return split_root / f"eval_iter{iter_n}"


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """将 Parquet 读取的 numpy array 字段统一转换为 Python 原生 list。

    Parquet/Arrow 格式会将嵌套列表存储为 Arrow 数组，pandas 读取后转换为
    numpy.ndarray。直接对这些对象做布尔判断或保存为 parquet 会出错。
    在数据进入 pipeline 前统一转换，从根本上解决类型问题。
    """
    def _to_native(val: Any) -> Any:
        if hasattr(val, "tolist"):
            return val.tolist()
        return val

    def _normalize_answer(ans: Any) -> Dict[str, Any]:
        if not isinstance(ans, dict):
            return {"correct_answers": [], "wrong_answers": []}
        ca = ans.get("correct_answers")
        wa = ans.get("wrong_answers")
        return {
            "correct_answers": _to_native(ca) if ca is not None else [],
            "wrong_answers": _to_native(wa) if wa is not None else [],
        }

    def _normalize_meta(meta: Any) -> Any:
        if not isinstance(meta, dict):
            return meta
        return {k: _to_native(v) for k, v in meta.items()}

    df = df.copy()
    if "answer" in df.columns:
        df["answer"] = df["answer"].apply(_normalize_answer)
    if "meta" in df.columns:
        df["meta"] = df["meta"].apply(_normalize_meta)
    return df


def load_split_df(
    split_file: Path,
    split_root: Path,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """加载单个 split 文件，应用采样，保存 sampled_input.parquet，返回规范化后的 DataFrame。

    采样结果持久化到 split_root/sampled_input.parquet，finalize 时直接读取，
    不再重新采样，避免行数不匹配问题。
    """
    df = pd.read_parquet(split_file)
    source = str(split_file)

    if config.get("sampling_enabled"):
        max_samples = config.get("max_samples", 100)
        if len(df) > max_samples:
            original_len = len(df)
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
            logger.info(f"[load_split] 采样 {max_samples}/{original_len} 条 from {split_file.name}")

    df = normalize_df(df)
    logger.info(f"[load_split] rows={len(df)} from {source}")

    # 持久化采样结果，供 finalize 使用
    split_root.mkdir(parents=True, exist_ok=True)
    sampled_path = split_root / "sampled_input.parquet"
    df.to_parquet(sampled_path, index=False)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 决策树标签分配
# ──────────────────────────────────────────────────────────────────────────────


def _assign_one(bucket: str, answerable: Optional[bool], is_shortcut: Optional[bool]) -> Tuple[str, Optional[str]]:
    """单条样本的决策树规则 → (label, repair_type)。"""
    if bucket == _BUCKET_ALL_PASSED:
        return _LABEL_EASY, "easy"

    if bucket == _BUCKET_ALL_FAILED:
        if answerable == True:  # noqa: E712
            return _LABEL_HARD, None
        # answerable False 或 None（解析失败也按 unanswerable 处理保守一些）
        return _LABEL_BAD, "unanswerable"

    # bucket == partial
    if answerable != True:  # noqa: E712
        return _LABEL_BAD, "unanswerable"
    if is_shortcut == True:  # noqa: E712
        return _LABEL_SHORTCUT, "shortcut"
    return _LABEL_MEDIUM, None


def assign_labels(
    input_df: pd.DataFrame,
    passk_df: pd.DataFrame,
    ans_df: pd.DataFrame,
    shortcut_df: pd.DataFrame,
    iter_n: int,
) -> pd.DataFrame:
    """合并三阶段输出 → labels DataFrame。

    通过 sample_id left join，保证：
      - 每条 input_df 对应一行 labels（行序与 input_df 一致）
      - 缺失的字段（all_passed 没跑 answerability 等）→ NaN/None

    Returns:
        DataFrame 列：sample_id, pass_at_k, bucket, answerable, answerability_label,
                     answerability_reason, no_story_pass, no_question_pass,
                     no_options_pass, is_shortcut, label, repair_type, iter
    """
    # 从 input_df 抽 sample_id 顺序作为基准
    base_ids: List[str] = []
    for idx, row in input_df.reset_index(drop=True).iterrows():
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        sid = meta.get("id") if isinstance(meta, dict) else None
        base_ids.append(str(sid) if sid else f"row_{idx}")
    base = pd.DataFrame({"sample_id": base_ids})

    # 检查 sample_id 是否有重复
    if base["sample_id"].duplicated().any():
        dup_count = base["sample_id"].duplicated().sum()
        logger.warning(f"[assign_labels] 检测到 {dup_count} 个重复的 sample_id，将使用位置索引")
        # 使用位置索引而非 sample_id merge
        base["__idx"] = range(len(base))
        passk_df = passk_df.copy()
        passk_df["__idx"] = range(len(passk_df))
        ans_df = ans_df.copy() if not ans_df.empty else ans_df
        if not ans_df.empty:
            ans_df["__idx"] = range(len(ans_df))
        shortcut_df = shortcut_df.copy() if not shortcut_df.empty else shortcut_df
        if not shortcut_df.empty:
            shortcut_df["__idx"] = range(len(shortcut_df))
        merge_key = "__idx"
    else:
        merge_key = "sample_id"

    # passk: sample_id, prompt_type, correct_letters, pass_at_k, bucket
    if not passk_df.empty:
        if merge_key == "__idx":
            base = base.merge(
                passk_df[["__idx", "pass_at_k", "bucket"]],
                on="__idx", how="left",
            )
        else:
            base = base.merge(
                passk_df[["sample_id", "pass_at_k", "bucket"]],
                on="sample_id", how="left",
            )
    else:
        base["pass_at_k"] = pd.NA
        base["bucket"] = pd.NA

    # answerability: sample_id, label, reason, answerable
    if not ans_df.empty:
        ans_renamed = ans_df.rename(columns={
            "label": "answerability_label",
            "reason": "answerability_reason",
        })
        if merge_key == "__idx":
            base = base.merge(
                ans_renamed[["__idx", "answerability_label", "answerability_reason", "answerable"]],
                on="__idx", how="left",
            )
        else:
            base = base.merge(
                ans_renamed[["sample_id", "answerability_label", "answerability_reason", "answerable"]],
                on="sample_id", how="left",
            )
    else:
        base["answerability_label"] = None
        base["answerability_reason"] = None
        base["answerable"] = None

    # shortcut: sample_id, no_story_pass, no_question_pass, no_options_pass, is_shortcut
    if not shortcut_df.empty:
        if merge_key == "__idx":
            base = base.merge(
                shortcut_df[["__idx", "no_story_pass", "no_question_pass",
                             "no_options_pass", "is_shortcut"]],
                on="__idx", how="left",
            )
        else:
            base = base.merge(
                shortcut_df[["sample_id", "no_story_pass", "no_question_pass",
                             "no_options_pass", "is_shortcut"]],
                on="sample_id", how="left",
            )
    else:
        base["no_story_pass"] = pd.NA
        base["no_question_pass"] = pd.NA
        base["no_options_pass"] = pd.NA
        base["is_shortcut"] = None

    # 清理临时索引列
    if merge_key == "__idx":
        base = base.drop(columns=["__idx"])

    # 决策树打标
    labels: List[str] = []
    repair_types: List[Optional[str]] = []
    for _, row in base.iterrows():
        bucket = row.get("bucket")
        answerable = row.get("answerable")
        is_sc = row.get("is_shortcut")

        # pandas NA / numpy array → Python scalar
        # 使用 == 而不是 is 来避免 numpy array 的布尔判断问题
        ans_clean: Optional[bool]
        try:
            if pd.notna(answerable) and answerable == True:  # noqa: E712
                ans_clean = True
            elif pd.notna(answerable) and answerable == False:  # noqa: E712
                ans_clean = False
            else:
                ans_clean = None
        except (ValueError, TypeError):
            # 如果是 array 或其他无法比较的类型，标记为 None
            ans_clean = None

        sc_clean: Optional[bool]
        try:
            if pd.notna(is_sc) and is_sc == True:  # noqa: E712
                sc_clean = True
            elif pd.notna(is_sc) and is_sc == False:  # noqa: E712
                sc_clean = False
            else:
                sc_clean = None
        except (ValueError, TypeError):
            sc_clean = None

        if not isinstance(bucket, str):
            # passk 缺失 → 标 bad（防御性，正常不该走到）
            labels.append(_LABEL_BAD)
            repair_types.append("unanswerable")
            continue
        lbl, rt = _assign_one(bucket, ans_clean, sc_clean)
        labels.append(lbl)
        repair_types.append(rt)

    base["label"] = labels
    base["repair_type"] = repair_types
    base["iter"] = iter_n
    return base


# ──────────────────────────────────────────────────────────────────────────────
# Summary 汇总
# ──────────────────────────────────────────────────────────────────────────────


def build_summary(labels_df: pd.DataFrame, iter_n: int) -> Dict[str, Any]:
    """统计标签分布。"""
    n = len(labels_df)
    counts: Dict[str, int] = {}
    if n > 0:
        vc = labels_df["label"].value_counts(dropna=False).to_dict()
        for k, v in vc.items():
            counts[str(k)] = int(v)
    pct = {k: round(v / n, 4) for k, v in counts.items()} if n > 0 else {}
    repair_count = 0
    if n > 0 and "repair_type" in labels_df.columns:
        repair_count = int(labels_df["repair_type"].notna().sum())
    return {
        "iter": iter_n,
        "total": n,
        "label_counts": counts,
        "label_pct": pct,
        "repair_count": repair_count,
    }


def _write_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# 单轮编排
# ──────────────────────────────────────────────────────────────────────────────


def _build_clients() -> Dict[str, Any]:
    """一次性构造所有 client（避免每轮重复连接）。"""
    answer = load_answer_models()
    return {
        "simple": answer["simple"],
        "strong": answer["strong"],
        "judge": load_judge_client("strong"),  # shortcut 的 no_options 维度判 open QA
        "repair": load_judge_client("strong"),  # repair 用 StructureClient
    }


def run_single_iteration(
    dataset: str,
    iter_n: int,
    input_df: pd.DataFrame,
    out_dir: Path,
    clients: Dict[str, Any],
    repair_enabled: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """一轮决策树评估 + 可选修复。

    Args:
        repair_enabled: False 时跳过 Phase E（用于最后一轮修复产出的纯评估）

    Returns:
        (labels_df, keep_df, repaired_df)
        - labels_df：本轮所有样本的标签
        - keep_df：label∈{hard,medium} 的原始样本
        - repaired_df：修复成功的样本（repair_enabled=False 时为空）
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if input_df.empty:
        logger.warning(f"[iter{iter_n}] input_df 为空，跳过该轮")
        empty = pd.DataFrame()
        return empty, empty, empty

    input_df = input_df.reset_index(drop=True)

    # ── Phase B：pass@k 全量 ────────────────────────────────────────────────
    passk_path = out_dir / "passk.parquet"
    if passk_path.exists():
        passk_df = pd.read_parquet(passk_path)
        logger.info(f"[iter{iter_n}] ⏭️  Phase B: 读取已有结果 ({len(passk_df)} rows)")
    else:
        logger.info(f"[iter{iter_n}] 📊 Phase B: pass@k 评估 (k={_PASS_K})")
        passk_df = run_passk_on_df(input_df, dataset, k=_PASS_K, simple_client=clients["simple"])
        write_passk_parquet(passk_df, passk_path)

    # ── Phase C：answerability（仅 partial + all_failed）─────────────────────
    ans_path = out_dir / "answerability.parquet"
    if ans_path.exists():
        ans_df = pd.read_parquet(ans_path)
        logger.info(f"[iter{iter_n}] ⏭️  Phase C: 读取已有结果 ({len(ans_df)} rows)")
    else:
        logger.info(f"[iter{iter_n}] 🔍 Phase C: answerability 判断")
        if _SKIP_ANSWERABILITY_ON_ALL_PASSED:
            ans_target_mask = passk_df["bucket"].isin([_BUCKET_PARTIAL, _BUCKET_ALL_FAILED])
        else:
            ans_target_mask = pd.Series([True] * len(passk_df))
        ans_target_idx = passk_df.index[ans_target_mask].tolist()
        if len(ans_target_idx) > 0:
            ans_subset = input_df.iloc[ans_target_idx].reset_index(drop=True)
            ans_df = run_answerability_on_df(ans_subset, dataset, strong_client=clients["strong"])
        else:
            ans_df = pd.DataFrame(columns=["sample_id", "label", "reason", "answerable"])
            logger.info(f"[iter{iter_n}] ⏭️  跳过 answerability (all_passed)")
        write_answerability_parquet(ans_df, ans_path)

    # ── Phase D：shortcut 三探测（仅 partial + answerable）────────────────────
    shortcut_path = out_dir / "shortcut.parquet"
    if shortcut_path.exists():
        shortcut_df = pd.read_parquet(shortcut_path)
        logger.info(f"[iter{iter_n}] ⏭️  Phase D: 读取已有结果 ({len(shortcut_df)} rows)")
    else:
        logger.info(f"[iter{iter_n}] 🎯 Phase D: shortcut 检测")
        if _SHORTCUT_ENABLED and not ans_df.empty:
            partial_ids = set(passk_df.loc[passk_df["bucket"] == _BUCKET_PARTIAL, "sample_id"])
            answerable_ids = set(ans_df.loc[ans_df["answerable"] == True, "sample_id"])  # noqa: E712
            target_ids = partial_ids & answerable_ids
            if len(target_ids) > 0:
                mask = passk_df["sample_id"].isin(target_ids)
                sc_subset = input_df.iloc[passk_df.index[mask].tolist()].reset_index(drop=True)
                shortcut_df = run_shortcut_on_df(
                    sc_subset,
                    dataset,
                    k=_PASS_K,
                    threshold=_SHORTCUT_THRESHOLD,
                    simple_client=clients["simple"],
                    judge_client=clients["judge"],
                    dimensions=_SHORTCUT_DIMENSIONS,
                )
            else:
                shortcut_df = pd.DataFrame(columns=[
                    "sample_id", "no_story_pass", "no_question_pass", "no_options_pass", "is_shortcut",
                ])
                logger.info(f"[iter{iter_n}] ⏭️  跳过 shortcut (无 partial+answerable)")
        else:
            shortcut_df = pd.DataFrame(columns=[
                "sample_id", "no_story_pass", "no_question_pass", "no_options_pass", "is_shortcut",
            ])
            logger.info(f"[iter{iter_n}] ⏭️  跳过 shortcut (已禁用或无数据)")
        write_shortcut_parquet(shortcut_df, shortcut_path)

    # ── 决策树打标 ──────────────────────────────────────────────────────────
    labels_path = out_dir / "labels.parquet"
    if labels_path.exists():
        labels_df = pd.read_parquet(labels_path)
        logger.info(f"[iter{iter_n}] ⏭️  决策树打标: 读取已有结果 ({len(labels_df)} rows)")
    else:
        logger.info(f"[iter{iter_n}] 🏷️  决策树打标")
        labels_df = assign_labels(input_df, passk_df, ans_df, shortcut_df, iter_n)
        labels_df.to_parquet(labels_path, index=False)

    if len(labels_df) != len(input_df):
        logger.error(
            f"[iter{iter_n}] labels_df 行数 ({len(labels_df)}) != input_df 行数 ({len(input_df)})"
        )
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # ── keep_df：hard + medium 的原始样本 ────────────────────────────────────
    keep_mask = labels_df["label"].isin(_KEEP_LABELS)
    keep_df = input_df.iloc[keep_mask.values].copy() if keep_mask.any() else pd.DataFrame(columns=input_df.columns)
    logger.info(f"[iter{iter_n}] ✅ 保留样本: {len(keep_df)} 条 (hard + medium)")

    # ── Phase E：repair ──────────────────────────────────────────────────────
    repaired_path = out_dir / "repaired.parquet"
    if repair_enabled:
        if repaired_path.exists():
            repaired_df = pd.read_parquet(repaired_path)
            repaired_df = normalize_df(repaired_df) if not repaired_df.empty else repaired_df
            logger.info(f"[iter{iter_n}] ⏭️  Phase E: 读取已有结果 ({len(repaired_df)} rows)")
        else:
            logger.info(f"[iter{iter_n}] 🔧 Phase E: 修复样本")
            repaired_df = repair_samples(
                input_df,
                labels_df,
                dataset,
                iter_n,
                repair_client=clients["repair"],
                enabled_types=_REPAIR_ENABLED_TYPES,
            )
            repaired_df = normalize_df(repaired_df) if not repaired_df.empty else repaired_df
            write_repaired_parquet(repaired_df, repaired_path)
    else:
        repaired_df = pd.DataFrame()
        logger.info(f"[iter{iter_n}] ⏭️  跳过 Phase E（纯评估轮）")

    # ── Summary ────────────────────────────────────────────────────────────
    summary = build_summary(labels_df, iter_n)
    summary["repair_ok"] = int(len(repaired_df))
    summary["keep_pool_count"] = int(len(keep_df))
    _write_json(summary, out_dir / "summary.json")
    logger.info(
        f"[iter{iter_n}] 📈 汇总: total={summary['total']} "
        f"labels={summary['label_counts']} repair_ok={summary['repair_ok']} keep={summary['keep_pool_count']}"
    )

    return labels_df, keep_df, repaired_df


# ──────────────────────────────────────────────────────────────────────────────
# 迭代循环
# ──────────────────────────────────────────────────────────────────────────────


def _mark_unfixable(out_dir: Path, labels_df: pd.DataFrame, iter_n: int) -> pd.DataFrame:
    """最后一轮仍需修复的样本 → label=unfixable，重写 labels.parquet + summary.json。"""
    unfixable_mask = labels_df["repair_type"].notna() & labels_df["label"].isin(
        [_LABEL_EASY, _LABEL_SHORTCUT, _LABEL_BAD]
    )
    if not unfixable_mask.any():
        return labels_df
    labels_df = labels_df.copy()
    labels_df.loc[unfixable_mask, "label"] = _LABEL_UNFIXABLE
    labels_df.to_parquet(out_dir / "labels.parquet", index=False)
    summary = build_summary(labels_df, iter_n)
    _write_json(summary, out_dir / "summary.json")
    logger.info(f"[iter{iter_n}] 最后一轮仍未修复 → 标 unfixable: {int(unfixable_mask.sum())}")
    return labels_df


def run_filter_loop(
    dataset: str,
    split_file: Path,
    max_iter: int,
    config: Dict[str, Any],
    clients: Dict[str, Any],
) -> Dict[str, Any]:
    """对单个 split 文件运行完整的迭代评估+修复循环。

    输出目录：{output_root}/{dataset}/split_{split_stem}/eval_iter{N}/

    Returns:
        汇总 dict {split, iters_run, label_counts_by_iter, total_keep}
    """
    if max_iter < 1:
        raise ValueError(f"max_iter 必须 >= 1，得到 {max_iter}")

    split_stem = split_file.stem
    split_root = _split_dir(config["output_root"], dataset, split_stem)

    logger.info(f"\n{'='*60}")
    logger.info(f"📂 [{dataset}] split={split_stem}")
    logger.info(f"{'='*60}")

    # 加载并采样，结果持久化到 sampled_input.parquet
    input_df = load_split_df(split_file, split_root, config)
    if input_df.empty:
        logger.warning(f"[{split_stem}] 输入为空，跳过")
        return {"split": split_stem, "iters_run": 0, "max_iter": max_iter,
                "label_counts_by_iter": [], "total_keep_pool": 0}

    iters_run = 0
    label_counts_by_iter: List[Dict[str, Any]] = []
    total_keep = 0
    last_labels_df: Optional[pd.DataFrame] = None
    last_out_dir: Optional[Path] = None
    current_input_df: Optional[pd.DataFrame] = input_df

    for iter_n in range(1, max_iter + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 [{dataset}/{split_stem}] iter={iter_n}/{max_iter}")
        logger.info(f"{'='*60}")

        out_dir = _iter_dir(split_root, iter_n)
        summary_path = out_dir / "summary.json"

        # iter 级断点：整轮已完成，跳过
        if summary_path.exists():
            with open(summary_path, encoding="utf-8") as _f:
                cached_summary = json.load(_f)
            iters_run += 1
            label_counts_by_iter.append({
                "iter": iter_n,
                "total": cached_summary.get("total", 0),
                "label_counts": cached_summary.get("label_counts", {}),
            })
            total_keep += int(cached_summary.get("keep_pool_count", 0))
            labels_path = out_dir / "labels.parquet"
            last_labels_df = pd.read_parquet(labels_path) if labels_path.exists() else pd.DataFrame()
            last_out_dir = out_dir
            repaired_path = out_dir / "repaired.parquet"
            if repaired_path.exists():
                _rep = normalize_df(pd.read_parquet(repaired_path))
                current_input_df = _rep if not _rep.empty else None
            else:
                current_input_df = None
            if current_input_df is None:
                logger.info(f"⏭️  [loop] iter={iter_n} 已完成（无修复产出），提前终止")
                break
            logger.info(f"⏭️  [loop] iter={iter_n} 已完成，跳过，传递 {len(current_input_df)} 条修复样本")
            continue

        if current_input_df is None or current_input_df.empty:
            logger.info(f"⚠️  [loop] iter={iter_n} 无输入，提前终止")
            break

        labels_df, keep_df, repaired_df = run_single_iteration(
            dataset, iter_n, current_input_df, out_dir, clients, repair_enabled=True,
        )
        iters_run += 1
        label_counts_by_iter.append({
            "iter": iter_n,
            "total": int(len(labels_df)),
            "label_counts": labels_df["label"].value_counts(dropna=False).to_dict() if len(labels_df) else {},
        })
        total_keep += int(len(keep_df))
        last_labels_df = labels_df
        last_out_dir = out_dir

        if repaired_df.empty:
            logger.info(f"✅ [loop] iter={iter_n} 无修复产出，提前终止")
            current_input_df = None
            break

        current_input_df = repaired_df
        logger.info(f"[iter{iter_n}] ➡️  传递 {len(repaired_df)} 条修复样本到下一轮")

    # 最后一轮仍需修复的 → unfixable
    if last_labels_df is not None and last_out_dir is not None and iters_run >= max_iter:
        last_labels_df = _mark_unfixable(last_out_dir, last_labels_df, iters_run)

    # 对最后一轮修复产出做额外纯评估（不修复）
    if current_input_df is not None and not current_input_df.empty:
        extra_iter = iters_run + 1
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 [{dataset}/{split_stem}] 额外纯评估轮 iter={extra_iter}（最后一轮修复产出）")
        logger.info(f"{'='*60}")
        out_dir = _iter_dir(split_root, extra_iter)
        labels_df, keep_df, _ = run_single_iteration(
            dataset, extra_iter, current_input_df, out_dir, clients, repair_enabled=False,
        )
        iters_run += 1
        label_counts_by_iter.append({
            "iter": extra_iter,
            "total": int(len(labels_df)),
            "label_counts": labels_df["label"].value_counts(dropna=False).to_dict() if len(labels_df) else {},
        })
        total_keep += int(len(keep_df))

    summary = {
        "split": split_stem,
        "iters_run": iters_run,
        "max_iter": max_iter,
        "label_counts_by_iter": label_counts_by_iter,
        "total_keep_pool": total_keep,
    }
    _write_json(summary, split_root / "filter_summary.json")
    logger.info(f"🎉 [{dataset}/{split_stem}] 完成! iters_run={iters_run} total_keep={total_keep}")
    return summary


def run_filter_loop_all_splits(
    dataset: str,
    max_iter: int,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """遍历数据集所有 split 文件，逐个运行 run_filter_loop。

    Returns:
        汇总 dict {dataset, splits_run, total_keep_pool}
    """
    dataset_dir = Path(config["input_root"]) / dataset
    split_files = sorted(dataset_dir.glob("*.parquet")) if dataset_dir.exists() else []
    if not split_files:
        raise FileNotFoundError(f"未找到 parquet 文件: {dataset_dir}")

    clients = _build_clients()
    splits_run = 0
    total_keep = 0
    split_summaries: List[Dict[str, Any]] = []

    for split_file in split_files:
        split_root = _split_dir(config["output_root"], dataset, split_file.stem)
        split_summary_path = split_root / "filter_summary.json"
        # split 级断点：整个 split 已完成，跳过
        if split_summary_path.exists():
            try:
                with open(split_summary_path, encoding="utf-8") as _f:
                    cached_split = json.load(_f)
                splits_run += 1
                total_keep += cached_split.get("total_keep_pool", 0)
                split_summaries.append(cached_split)
                logger.info(f"⏭️  [{split_file.stem}] 已完成，跳过")
                continue
            except Exception:
                pass  # 读取失败则重新跑该 split
        try:
            split_summary = run_filter_loop(dataset, split_file, max_iter, config, clients)
            splits_run += 1
            total_keep += split_summary.get("total_keep_pool", 0)
            split_summaries.append(split_summary)
        except Exception as e:
            import traceback
            logger.error(f"[{dataset}/{split_file.stem}] 处理失败: {e}")
            logger.error(traceback.format_exc())

    summary = {
        "dataset": dataset,
        "splits_run": splits_run,
        "total_splits": len(split_files),
        "total_keep_pool": total_keep,
        "split_summaries": split_summaries,
    }
    overall_path = Path(config["output_root"]) / dataset / "filter_summary.json"
    _write_json(summary, overall_path)
    logger.info(f"\n{'='*60}")
    logger.info(f"🎉 [{dataset}] 全部完成! splits={splits_run}/{len(split_files)} total_keep={total_keep}")
    logger.info(f"{'='*60}\n")
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Finalize：合并所有轮 hard+medium → train_set.parquet
# ──────────────────────────────────────────────────────────────────────────────


def run_finalize(dataset: str, config: Dict[str, Any]) -> Path:
    """跨所有 split、所有轮次收集 hard+medium 样本 → datasets/train_set.parquet。

    每个 split 的每一轮：
    - iter=1 的原始数据从 split_root/sampled_input.parquet 读取
    - iter>1 的原始数据从 split_root/eval_iter{N-1}/repaired.parquet 读取
    同一 sample_id 跨轮出现时保留最后一次（最新修复版本）。
    """
    ds_root = Path(config["output_root"]) / "_intermediate" / dataset
    if not ds_root.exists():
        raise FileNotFoundError(f"未找到评估输出目录: {ds_root}")

    split_dirs = sorted([p for p in ds_root.glob("split_*") if p.is_dir()])
    if not split_dirs:
        raise FileNotFoundError(f"{ds_root} 下未找到 split_* 目录")

    keep_frames: List[pd.DataFrame] = []

    for split_root in split_dirs:
        iter_dirs = sorted(
            [p for p in split_root.glob("eval_iter*") if p.is_dir()],
            key=lambda p: int(p.name.replace("eval_iter", "") or 0),
        )
        for iter_dir in iter_dirs:
            labels_path = iter_dir / "labels.parquet"
            if not labels_path.exists():
                continue
            labels_df = pd.read_parquet(labels_path)
            if labels_df.empty:
                continue

            iter_n = int(iter_dir.name.replace("eval_iter", ""))

            # 加载对应的原始输入（不重新采样）
            if iter_n == 1:
                input_path = split_root / "sampled_input.parquet"
            else:
                input_path = split_root / f"eval_iter{iter_n - 1}" / "repaired.parquet"

            if not input_path.exists():
                logger.warning(f"[finalize] 找不到输入文件，跳过: {input_path}")
                continue

            input_df = pd.read_parquet(input_path)
            input_df = normalize_df(input_df)  # 规范化数据
            if len(input_df) != len(labels_df):
                logger.warning(
                    f"[finalize] {iter_dir.name} input/labels 行数不一致，跳过: "
                    f"input={len(input_df)} labels={len(labels_df)}"
                )
                continue

            labels_df = labels_df.reset_index(drop=True)
            input_df = input_df.reset_index(drop=True)
            keep_mask = labels_df["label"].isin(_KEEP_LABELS)
            if not keep_mask.any():
                continue

            keep_rows = input_df.loc[keep_mask.values].copy()
            keep_rows["__sample_id"] = labels_df.loc[keep_mask, "sample_id"].values
            keep_rows["__iter"] = iter_n
            keep_frames.append(keep_rows)

    # 最终数据集保存到显眼的 datasets/ 目录
    datasets_root = Path(config["output_root"]) / "datasets"
    datasets_root.mkdir(parents=True, exist_ok=True)
    out_path = datasets_root / f"{dataset}_filtered.parquet"

    if not keep_frames:
        pd.DataFrame().to_parquet(out_path, index=False)
        logger.warning(f"[finalize] {dataset} 无 hard/medium 样本，写空 parquet → {out_path}")
        return out_path

    merged = pd.concat(keep_frames, ignore_index=True)
    merged = merged.drop_duplicates(subset="__sample_id", keep="last")
    merged = merged.drop(columns=["__sample_id", "__iter"], errors="ignore").reset_index(drop=True)
    merged.to_parquet(out_path, index=False)
    logger.info(f"[finalize] {dataset} → {out_path} rows={len(merged)}")
    return out_path


__all__ = [
    "load_filter_config",
    "normalize_df",
    "load_split_df",
    "assign_labels",
    "build_summary",
    "run_single_iteration",
    "run_filter_loop",
    "run_filter_loop_all_splits",
    "run_finalize",
]
