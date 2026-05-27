"""
Stage 4：LSH 守门员过滤

读取 Stage 3 输出的 synth_raw/candidates_iter*.jsonl，
通过 LSH MinHash 去除与测试集相似的样本，
通过的样本写入 synth_clean/*.parquet。
"""

import glob
import json
import logging
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── 超参数 ────────────────────────────────────────────────────────────────────
_NGRAM = 4
_NUM_PERM = 128
_THRESHOLD = 0.6

_KNOWN_TASKS = [
    "BigToM", "EmoBench", "FanToM", "HiToM", "SimpleToM", "SocialIQA", "ToMBench"
]


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


def _char_ngrams(text: str) -> Set[str]:
    text = _normalize(text)
    if len(text) < _NGRAM:
        return {text} if text else set()
    return {text[i: i + _NGRAM] for i in range(len(text) - _NGRAM + 1)}


def _make_minhash(text: str):
    from datasketch import MinHash
    m = MinHash(num_perm=_NUM_PERM)
    for g in _char_ngrams(text):
        m.update(g.encode("utf-8"))
    return m


# ── 索引构建 ──────────────────────────────────────────────────────────────────

def build_test_index(test_root: Path, tasks: Optional[List[str]] = None):
    """对所有 test parquet 构建 MinHashLSH 索引。

    Returns:
        (lsh, grams_map): lsh 用于 query，grams_map 用于精确 Jaccard 验证。
    """
    from datasketch import MinHashLSH
    import pyarrow.parquet as pq

    if tasks is None:
        tasks = _KNOWN_TASKS

    lsh = MinHashLSH(threshold=_THRESHOLD, num_perm=_NUM_PERM)
    grams_map: Dict[str, FrozenSet] = {}
    total = 0

    for task in tasks:
        parquet_files = sorted(glob.glob(str(test_root / task / "*.parquet")))
        if not parquet_files:
            logger.warning(f"  No test parquet for {task}: {test_root / task}")
            continue
        for pfile in parquet_files:
            rows = pq.read_table(pfile).to_pylist()
            for row in rows:
                text = _build_text(row)
                if not text:
                    continue
                meta = row.get("meta", {}) or {}
                row_id = f"{task}::{meta.get('id', total)}"
                if row_id in grams_map:
                    row_id = f"{row_id}__dup{total}"
                mh = _make_minhash(text)
                lsh.insert(row_id, mh)
                grams_map[row_id] = frozenset(_char_ngrams(text))
                total += 1
        logger.info(f"  {task}: indexed")

    logger.info(f"Test index built: {total} samples from {len(tasks)} tasks")
    return lsh, grams_map


# ── 过滤逻辑 ──────────────────────────────────────────────────────────────────

def _is_leaked(candidate: dict, lsh, grams_map: Dict[str, FrozenSet]) -> Tuple[bool, Optional[str]]:
    text = _build_text(candidate)
    if not text:
        return False, None
    mh = _make_minhash(text)
    hits = lsh.query(mh)
    if not hits:
        return False, None
    cand_grams = frozenset(_char_ngrams(text))
    for hit in hits:
        test_grams = grams_map.get(hit, frozenset())
        if not test_grams:
            continue
        union = len(cand_grams | test_grams)
        j = len(cand_grams & test_grams) / union if union else 0.0
        if j >= _THRESHOLD:
            return True, f"{hit} (jaccard={j:.3f})"
    return False, None


def filter_candidates(
    candidates: List[dict],
    lsh,
    grams_map: Dict[str, FrozenSet],
) -> Tuple[List[dict], List[dict]]:
    """返回 (passed, dropped)，dropped 里每条带 _drop_reason 字段。"""
    passed, dropped = [], []
    for cand in candidates:
        leaked, reason = _is_leaked(cand, lsh, grams_map)
        if leaked:
            cand = dict(cand)
            cand["_drop_reason"] = reason
            dropped.append(cand)
        else:
            passed.append(cand)
    return passed, dropped


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
    iteration: int = 1,
) -> Dict[str, Dict[str, int]]:
    """对 synth_raw 下所有 candidates_iter*.jsonl 跑 LSH 守门员过滤。

    Returns:
        dict: {dataset_name: {"synth_clean": N, "synth_dropped": M}}
    """
    test_root = config.get("leakage_guard", {}).get("test_root", "test_data")
    abs_test_root = Path(test_root)
    if not abs_test_root.is_absolute():
        abs_test_root = (Path(__file__).parent.parent / test_root).resolve()

    lsh, grams_map = build_test_index(abs_test_root)

    synth_raw_root = output_path / "synth_raw"
    synth_clean_root = output_path / "synth_clean"
    all_stats: Dict[str, Dict[str, int]] = {}

    for ds_info in synthesis_datasets:
        ds = ds_info["name"]
        ds_raw_dir = synth_raw_root / ds
        if not ds_raw_dir.exists():
            continue

        for cand_file in sorted(ds_raw_dir.glob(f"candidates_iter{iteration}_*.jsonl")):
            candidates = []
            with open(cand_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        candidates.append(json.loads(line))

            if not candidates:
                continue

            passed, dropped = filter_candidates(candidates, lsh, grams_map)
            rate = len(passed) / max(1, len(candidates))
            logger.info(
                f"  {ds}/{cand_file.name}: "
                f"{len(passed)}/{len(candidates)} passed ({rate:.1%}), {len(dropped)} dropped"
            )

            stem = cand_file.stem.replace("candidates", "synthetic")
            out_pass = synth_clean_root / ds / f"{stem}.parquet"
            out_drop = synth_clean_root / ds / f"{stem}_dropped.jsonl"
            log_path = synth_clean_root / ds / f"{stem}_dedupe_log.jsonl"

            save_as_parquet(passed, out_pass)
            if dropped:
                save_as_jsonl(dropped, out_drop)

            log_entries = [
                {
                    "file": str(cand_file), "total": len(candidates),
                    "passed": len(passed), "dropped": len(dropped),
                    "drop_rate": round(1 - rate, 4),
                }
            ] + [
                {"dropped_id": r.get("meta", {}).get("id", ""), "reason": r.get("_drop_reason", "")}
                for r in dropped
            ]
            save_as_jsonl(log_entries, log_path)

            all_stats.setdefault(ds, {})["synth_clean"] = len(passed)
            all_stats.setdefault(ds, {})["synth_dropped"] = len(dropped)

    return all_stats
