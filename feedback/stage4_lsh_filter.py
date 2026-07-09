"""
Stage 4：LSH 守门员过滤

读取 Stage 3 输出的 synth_raw/candidates_*.jsonl，
通过 LSH MinHash 去除与测试集相似的样本，
通过的样本写入 synth_clean/*.parquet。
"""

import glob
import json
import logging
import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from tqdm import tqdm

logger = logging.getLogger(__name__)

_KNOWN_TASKS = [
    "BigToM", "EmoBench", "FanToM", "HiToM", "SocialIQA", "ToMBench", "SoMBench"
]
_CACHE_VERSION = 1


# ── 文本工具 ──────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return " ".join(str(text).split())


def _build_text(row: dict) -> str:
    story = row.get("story", "") or ""
    if isinstance(story, dict):
        story = story.get("full_story", "") or " ".join(str(v) for v in story.values())
    question = row.get("question", "") or ""
    answer = row.get("answer", {}) or {}
    if isinstance(answer, dict):
        all_opts = list(answer.get("correct_answers", []) or []) + list(answer.get("wrong_answers", []) or [])
    else:
        all_opts = []
    parts = [_normalize(story), _normalize(question)] + [_normalize(o) for o in all_opts]
    return " ".join(p for p in parts if p)


def _char_ngrams(text: str, ngram: int = 4) -> Set[str]:
    text = _normalize(text)
    if len(text) < ngram:
        return {text} if text else set()
    return {text[i: i + ngram] for i in range(len(text) - ngram + 1)}


def _make_minhash(text: str, num_perm: int = 128, ngram: int = 4):
    from datasketch import MinHash
    m = MinHash(num_perm=num_perm)
    for g in _char_ngrams(text, ngram):
        m.update(g.encode("utf-8"))
    return m


def _collect_test_index_meta(test_root: Path, tasks: List[str], threshold: float = 0.6, ngram: int = 4, num_perm: int = 128) -> Dict[str, Any]:
    files: List[Dict[str, Any]] = []
    for task in tasks:
        for pfile in sorted(glob.glob(str(test_root / task / "*.parquet"))):
            p = Path(pfile)
            try:
                stat = p.stat()
            except FileNotFoundError:
                continue
            files.append(
                {
                    "path": str(p.resolve()),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )

    meta = {
        "version": _CACHE_VERSION,
        "test_root": str(test_root.resolve()),
        "tasks": list(tasks),
        "threshold": threshold,
        "ngram": ngram,
        "num_perm": num_perm,
        "files": files,
    }
    meta["signature"] = hashlib.sha256(
        json.dumps(meta, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return meta


def _load_cached_test_index(cache_path: Path, signature: str):
    try:
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            return None
        if payload.get("cache_meta", {}).get("signature") != signature:
            return None
        lsh = payload.get("lsh")
        grams_map = payload.get("grams_map")
        if lsh is None or grams_map is None:
            return None
        logger.info(f"Loaded cached test index from {cache_path}")
        return lsh, grams_map
    except Exception as e:
        logger.warning(f"Failed to load cached test index from {cache_path}: {e}")
        return None


def _save_cached_test_index(cache_path: Path, lsh, grams_map: Dict[str, FrozenSet], cache_meta: Dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_meta": cache_meta,
        "lsh": lsh,
        "grams_map": grams_map,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Built and cached test index to {cache_path}")


# ── 索引构建 ──────────────────────────────────────────────────────────────────

def build_test_index(
    test_root: Path,
    tasks: Optional[List[str]] = None,
    cache_dir: Optional[Path] = None,
    threshold: float = 0.6,
    num_perm: int = 128,
    ngram: int = 4,
):
    """对所有 test parquet 构建 MinHashLSH 索引。

    Returns:
        (lsh, grams_map): lsh 用于 query，grams_map 用于精确 Jaccard 验证。
    """
    from datasketch import MinHashLSH
    import pyarrow.parquet as pq

    if tasks is None:
        tasks = _KNOWN_TASKS

    cache_meta = _collect_test_index_meta(test_root, tasks, threshold=threshold, ngram=ngram, num_perm=num_perm)
    cache_path = None if cache_dir is None else cache_dir / f"test_index_{cache_meta['signature']}.pkl"
    if cache_path is not None and cache_path.exists():
        cached = _load_cached_test_index(cache_path, cache_meta["signature"])
        if cached is not None:
            return cached

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    grams_map: Dict[str, FrozenSet] = {}
    total = 0

    for task in tasks:
        parquet_files = sorted(glob.glob(str(test_root / task / "*.parquet")))
        if not parquet_files:
            logger.warning(f"  No test parquet for {task}: {test_root / task}")
            continue
        for pfile in parquet_files:
            rows = pq.read_table(pfile).to_pylist()
            for row in tqdm(rows, desc=f"{task}", leave=False):
                text = _build_text(row)
                if not text:
                    continue
                meta = row.get("meta", {}) or {}
                row_id = f"{task}::{meta.get('id', total)}"
                if row_id in grams_map:
                    row_id = f"{row_id}__dup{total}"
                mh = _make_minhash(text, num_perm=num_perm, ngram=ngram)
                lsh.insert(row_id, mh)
                grams_map[row_id] = frozenset(_char_ngrams(text, ngram))
                total += 1
        logger.info(f"  {task}: indexed")

    logger.info(f"Test index built: {total} samples from {len(tasks)} tasks")
    if cache_path is not None:
        try:
            _save_cached_test_index(cache_path, lsh, grams_map, cache_meta)
        except Exception as e:
            logger.warning(f"Failed to cache test index to {cache_path}: {e}")
    return lsh, grams_map


# ── 过滤逻辑 ──────────────────────────────────────────────────────────────────

def _is_leaked(
    candidate: dict,
    lsh,
    grams_map: Dict[str, FrozenSet],
    threshold: float = 0.6,
    num_perm: int = 128,
    ngram: int = 4,
) -> Tuple[bool, Optional[str]]:
    text = _build_text(candidate)
    if not text:
        return False, None
    mh = _make_minhash(text, num_perm=num_perm, ngram=ngram)
    hits = lsh.query(mh)
    if not hits:
        return False, None
    cand_grams = frozenset(_char_ngrams(text, ngram))
    for hit in hits:
        test_grams = grams_map.get(hit, frozenset())
        if not test_grams:
            continue
        union = len(cand_grams | test_grams)
        j = len(cand_grams & test_grams) / union if union else 0.0
        if j >= threshold:
            return True, f"{hit} (jaccard={j:.3f})"
    return False, None


def filter_candidates(
    candidates: List[dict],
    lsh,
    grams_map: Dict[str, FrozenSet],
    threshold: float = 0.6,
    num_perm: int = 128,
    ngram: int = 4,
) -> Tuple[List[dict], List[dict]]:
    """返回 (passed, dropped)，dropped 里每条带 _drop_reason 字段。"""
    passed, dropped = [], []
    for cand in candidates:
        leaked, reason = _is_leaked(cand, lsh, grams_map, threshold=threshold, num_perm=num_perm, ngram=ngram)
        if leaked:
            cand = dict(cand)
            cand["_drop_reason"] = reason
            dropped.append(cand)
        else:
            passed.append(cand)
    return passed, dropped


def deduplicate_internal(candidates: List[dict], threshold: float = 0.85, num_perm: int = 128, ngram: int = 4) -> Tuple[List[dict], List[dict]]:
    """内部去重：检测合成数据之间的相似样本

    Args:
        candidates: 候选样本列表
        threshold: Jaccard 相似度阈值（默认 0.85，比测试集去重更严格）
        num_perm: MinHash 排列数
        ngram: n-gram 大小

    Returns:
        (unique, duplicates) 元组，duplicates 里每条带 _drop_reason 字段
    """
    from datasketch import MinHash

    if not candidates:
        return [], []

    unique: List[dict] = []
    duplicates: List[dict] = []
    seen_hashes: List[Tuple[MinHash, FrozenSet[str], str]] = []  # (minhash, grams, sample_id)

    for cand in tqdm(candidates, desc="Internal dedup", leave=False):
        text = _build_text(cand)
        if not text:
            unique.append(cand)
            continue

        mh = _make_minhash(text, num_perm=num_perm, ngram=ngram)
        cand_grams = frozenset(_char_ngrams(text, ngram))
        cand_id = cand.get("meta", {}).get("id", "unknown")

        # 检查是否与已有样本重复
        is_dup = False
        for seen_mh, seen_grams, seen_id in seen_hashes:
            # 先用 MinHash 快速筛选
            if mh.jaccard(seen_mh) < threshold:
                continue
            # 精确 Jaccard 验证
            union = len(cand_grams | seen_grams)
            j = len(cand_grams & seen_grams) / union if union else 0.0
            if j >= threshold:
                cand_dup = dict(cand)
                cand_dup["_drop_reason"] = f"duplicate_of_{seen_id} (jaccard={j:.3f})"
                duplicates.append(cand_dup)
                is_dup = True
                break

        if not is_dup:
            unique.append(cand)
            seen_hashes.append((mh, cand_grams, cand_id))

    return unique, duplicates


# ── 保存 ─────────────────────────────────────────────────────────────────────

def save_as_parquet(rows: List[dict], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        logger.warning(f"  No rows to save to {path}")
        return
    clean_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    pq.write_table(pa.Table.from_pylist(clean_rows), str(path))
    logger.info(f"  Saved {len(rows)} rows to {path}")


def save_as_jsonl(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"  Saved {len(rows)} rows to {path}")


# ── 主入口 ────────────────────────────────────────────────────────────────────

def run_stage4_lsh_filter(
    config: Dict[str, Any],
    synthesis_datasets: List[Dict[str, Any]],
    output_path: Path,
) -> Dict[str, Dict[str, int]]:
    """对 synth_raw 下所有 candidates_*.jsonl 跑双重去重：
    1. 测试集去重（LSH + 精确 Jaccard，threshold=0.6）
    2. 内部去重（MinHash + 精确 Jaccard，threshold=0.85）

    V3 输出契约：
    - 中间产物写入 _intermediate/synth_raw/ 和 _intermediate/synth_clean/
    - 最终数据集写入 datasets/<DS>.parquet（显眼！）

    Returns:
        dict: {dataset_name: {"synth_clean": N, "synth_dropped_test": M, "synth_dropped_internal": K}}
    """
    # 从 config 读取 LSH 参数
    leakage_guard = config.get("leakage_guard", {})
    test_root = leakage_guard.get("test_root", "test_data")
    threshold = leakage_guard.get("threshold", 0.6)
    internal_threshold = leakage_guard.get("internal_threshold", 0.85)
    num_perm = leakage_guard.get("num_perm", 128)
    ngram = leakage_guard.get("ngram", 4)

    abs_test_root = Path(test_root)
    if not abs_test_root.is_absolute():
        abs_test_root = (Path(__file__).parent.parent / test_root).resolve()

    lsh, grams_map = build_test_index(
        abs_test_root,
        cache_dir=output_path / "_intermediate" / "cache" / "stage4_lsh",
        threshold=threshold,
        num_perm=num_perm,
        ngram=ngram,
    )

    synth_raw_root = output_path / "_intermediate" / "synth_raw"
    synth_clean_root = output_path / "_intermediate" / "synth_clean"
    datasets_root = output_path / "datasets"
    all_stats: Dict[str, Dict[str, int]] = {}

    for ds_info in synthesis_datasets:
        ds = ds_info["name"]
        ds_raw_dir = synth_raw_root / ds
        if not ds_raw_dir.exists():
            continue

        # 收集所有候选文件
        cand_files = sorted(ds_raw_dir.glob("candidates_*.jsonl"))
        if not cand_files:
            continue

        # 先合并所有候选样本
        all_candidates: List[dict] = []
        for cand_file in cand_files:
            with open(cand_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_candidates.append(json.loads(line))

        if not all_candidates:
            continue

        logger.info(f"  {ds}: {len(all_candidates)} raw candidates from {len(cand_files)} files")

        # 第一步：测试集去重
        passed_test, dropped_test = filter_candidates(
            all_candidates, lsh, grams_map,
            threshold=threshold, num_perm=num_perm, ngram=ngram
        )
        test_drop_rate = len(dropped_test) / max(1, len(all_candidates))
        logger.info(
            f"  {ds}: test dedup: {len(passed_test)}/{len(all_candidates)} passed "
            f"({1-test_drop_rate:.1%}), {len(dropped_test)} dropped"
        )

        # 第二步：内部去重
        passed_internal, dropped_internal = deduplicate_internal(
            passed_test, threshold=internal_threshold, num_perm=num_perm, ngram=ngram
        )
        internal_drop_rate = len(dropped_internal) / max(1, len(passed_test))
        logger.info(
            f"  {ds}: internal dedup: {len(passed_internal)}/{len(passed_test)} unique "
            f"({1-internal_drop_rate:.1%}), {len(dropped_internal)} duplicates"
        )

        # 保存去重日志到中间目录（按文件拆分，便于追溯）
        for cand_file in cand_files:
            stem = cand_file.stem.replace("candidates", "synthetic")

            # 测试集去重日志
            test_drop_path = synth_clean_root / ds / f"{stem}_dropped_test.jsonl"
            file_dropped_test = [
                d for d in dropped_test
                if d.get("meta", {}).get("id", "").startswith(stem.split("_")[0])
            ]
            if file_dropped_test:
                save_as_jsonl(file_dropped_test, test_drop_path)

            # 内部去重日志
            internal_drop_path = synth_clean_root / ds / f"{stem}_dropped_internal.jsonl"
            file_dropped_internal = [
                d for d in dropped_internal
                if d.get("meta", {}).get("id", "").startswith(stem.split("_")[0])
            ]
            if file_dropped_internal:
                save_as_jsonl(file_dropped_internal, internal_drop_path)

        # 汇总去重日志
        summary_log_path = synth_clean_root / ds / f"dedupe_summary.jsonl"
        summary_entries = [
            {
                "total_raw": len(all_candidates),
                "passed_test_dedup": len(passed_test),
                "dropped_test_dedup": len(dropped_test),
                "test_drop_rate": round(test_drop_rate, 4),
                "passed_internal_dedup": len(passed_internal),
                "dropped_internal_dedup": len(dropped_internal),
                "internal_drop_rate": round(internal_drop_rate, 4),
                "final_clean": len(passed_internal),
            }
        ]
        save_as_jsonl(summary_entries, summary_log_path)

        # 保存最终数据集到显眼位置
        final_path = datasets_root / f"{ds}.parquet"
        save_as_parquet(passed_internal, final_path)
        logger.info(
            f"  {ds}: final dataset: {len(passed_internal)}/{len(all_candidates)} "
            f"({len(passed_internal)/max(1,len(all_candidates)):.1%}) → {final_path}"
        )

        all_stats[ds] = {
            "synth_clean": len(passed_internal),
            "synth_dropped_test": len(dropped_test),
            "synth_dropped_internal": len(dropped_internal),
        }

    return all_stats
