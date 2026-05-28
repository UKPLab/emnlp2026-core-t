#!/usr/bin/env python3
"""
Table augmentor.

This script runs after table selection. For each query it reads the dense retriever
top-K tables, maps ``results_table_selection/selected_tables.json`` into that pool,
applies **initial table augmentation** (compat-based neighbor addition)), then **seeded expansion** (greedy growth by compatibility and
retrieval scores). It writes ``results_table_augmentation/`` outputs: augmented
tables, scores, post-initial-augmentation tables, per-query details, and optional
``eval_summary.json``.

Usage:
    python table_augmentor.py --dataset bird --llm-model "huggingface:Qwen/Qwen2.5-7B-Instruct" --embedding-model "fireworks:WhereIsAI/UAE-Large-V1"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv


def _project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "data").exists():
            return parent
    return current.parent


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from utils import DatabaseType, get_results_run_dir  # noqa: E402

TABLE_KEY_SEP = "#sep#"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _normalize_table_id(table_id: Any) -> str:
    s = str(table_id or "").strip()
    if TABLE_KEY_SEP in s:
        dbp, tp = s.split(TABLE_KEY_SEP, 1)
        return f"{dbp.strip().lower()}{TABLE_KEY_SEP}{tp.strip().lower()}"
    return s.lower()


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _metrics_for_one(pred: Sequence[str], gold: Sequence[str]) -> Dict[str, Any]:
    pred_set = {_normalize_table_id(x) for x in (pred or []) if _normalize_table_id(x)}
    gold_set = {_normalize_table_id(x) for x in (gold or []) if _normalize_table_id(x)}
    tp_set = pred_set.intersection(gold_set)
    tp = len(tp_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = _safe_div(tp, len(pred_set))
    recall = _safe_div(tp, len(gold_set))
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    perfect_recall = bool(gold_set.issubset(pred_set)) if gold_set else True
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "perfect_recall": perfect_recall,
        "pred_tables": sorted(pred_set),
        "gold_tables": sorted(gold_set),
        "matched_tables": sorted(tp_set),
        "missing_gold_tables": sorted(gold_set - pred_set),
        "extra_pred_tables": sorted(pred_set - gold_set),
    }


def _evaluate(preds: List[List[str]], golds: List[List[str]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    n = min(len(preds), len(golds))
    per_query: List[Dict[str, Any]] = []
    sum_tp = 0
    sum_fp = 0
    sum_fn = 0
    sum_p = 0.0
    sum_r = 0.0
    sum_f1 = 0.0
    sum_perfect = 0

    for i in range(n):
        m = _metrics_for_one(preds[i], golds[i])
        m["index"] = i
        per_query.append(m)
        sum_tp += int(m["tp"])
        sum_fp += int(m["fp"])
        sum_fn += int(m["fn"])
        sum_p += float(m["precision"])
        sum_r += float(m["recall"])
        sum_f1 += float(m["f1"])
        sum_perfect += 1 if m["perfect_recall"] else 0

    micro_precision = _safe_div(sum_tp, (sum_tp + sum_fp))
    micro_recall = _safe_div(sum_tp, (sum_tp + sum_fn))
    micro_f1 = (
        (2 * micro_precision * micro_recall / (micro_precision + micro_recall))
        if (micro_precision + micro_recall)
        else 0.0
    )
    summary = {
        "n_evaluated": n,
        "macro": {
            "precision": _safe_div(sum_p, n),
            "recall": _safe_div(sum_r, n),
            "f1": _safe_div(sum_f1, n),
            "perfect_recall": _safe_div(sum_perfect, n),
        },
        "micro": {
            "tp": sum_tp,
            "fp": sum_fp,
            "fn": sum_fn,
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
        },
    }
    return summary, per_query


def _safe_min(values: List[float]) -> float:
    return float(min(values)) if values else 0.0


def _safe_max(values: List[float]) -> float:
    return float(max(values)) if values else 0.0


def _safe_avg(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _value_counts_full_range(values: List[float]) -> Dict[str, int]:
    if not values:
        return {"0": 0}
    ints = [int(round(v)) for v in values]
    lo = min(ints)
    hi = max(ints)
    counts: Dict[str, int] = {str(i): 0 for i in range(lo, hi + 1)}
    for v in ints:
        counts[str(v)] = counts.get(str(v), 0) + 1
    return counts


def _load_gold_tables(path: Path) -> List[List[str]]:
    try:
        payload = _read_json(path)
    except Exception as e:
        print(f"Failed to read gold file '{path}': {e}")
        return []
    if not isinstance(payload, list):
        return []
    out: List[List[str]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            out.append([])
            continue
        db_id = str(entry.get("db_id", "") or "").strip().lower()
        tables = entry.get("tables", []) or []
        row: List[str] = []
        if isinstance(tables, list):
            for t in tables:
                tname = str(t or "").strip().lower()
                if not tname:
                    continue
                row.append(
                    _normalize_table_id(f"{db_id}{TABLE_KEY_SEP}{tname}") if db_id else _normalize_table_id(tname)
                )
        out.append(row)
    return out


def _load_queries(path: Path) -> List[Dict[str, Any]]:
    try:
        payload = _read_json(path)
    except Exception as e:
        print(f"Failed to read queries '{path}': {e}")
        return []
    return payload if isinstance(payload, list) else []


def _load_retrieved_tables(path: Path) -> List[List[str]]:
    try:
        payload = _read_json(path)
    except Exception as e:
        print(f"Failed to read dense retrieval data '{path}': {e}")
        return []
    if not isinstance(payload, list):
        return []
    out: List[List[str]] = []
    for entry in payload:
        if not isinstance(entry, list):
            out.append([])
            continue
        row: List[str] = []
        for item in entry:
            if isinstance(item, str):
                row.append(_normalize_table_id(item))
            elif isinstance(item, dict) and "table" in item:
                row.append(_normalize_table_id(item.get("table")))
        out.append([x for x in row if x])
    return out


def _load_selected_tables(path: Path) -> List[List[str]]:
    try:
        payload = _read_json(path)
    except Exception as e:
        print(f"Failed to read selected tables '{path}': {e}")
        return []
    if not isinstance(payload, list):
        return []
    out: List[List[str]] = []
    for row in payload:
        if not isinstance(row, list):
            out.append([])
            continue
        out.append([_normalize_table_id(x) for x in row if _normalize_table_id(x)])
    return out


def _load_retrieved_scores(path: Optional[Path]) -> Optional[List[List[Dict[str, Any]]]]:
    if path is None:
        return None
    try:
        payload = _read_json(path)
    except Exception:
        return None
    if not isinstance(payload, list):
        return None
    out: List[List[Dict[str, Any]]] = []
    for row in payload:
        if isinstance(row, list):
            out.append([x for x in row if isinstance(x, dict)])
        else:
            out.append([])
    return out


def _compatibility_path(compat_dir: Path, table_a: str, table_b: str) -> Optional[Path]:
    a = _normalize_table_id(table_a)
    b = _normalize_table_id(table_b)
    p1 = (compat_dir / f"{a}-{b}.json").resolve()
    p2 = (compat_dir / f"{b}-{a}.json").resolve()
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    return None


def _load_pair_compatibility(compat_dir: Path, table_a: str, table_b: str) -> float:
    path = _compatibility_path(compat_dir, table_a, table_b)
    if path is None:
        return 0.0
    try:
        payload = _read_json(path)
    except Exception:
        return 0.0
    try:
        return float((payload or {}).get("overall_compatibility", 0.0))
    except Exception:
        return 0.0


def _build_compat_matrix(compat_dir: Path, table_ids: Sequence[str]) -> List[List[float]]:
    n = len(table_ids)
    mat: List[List[float]] = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = _load_pair_compatibility(compat_dir, table_ids[i], table_ids[j])
            mat[i][j] = s
            mat[j][i] = s
    return mat


def _pair_stats(indices: Sequence[int], mat: List[List[float]]) -> Tuple[float, float]:
    if len(indices) <= 1:
        return (0.0, 0.0)
    vals: List[float] = []
    pos = 0
    all_pairs = 0
    for i_pos in range(len(indices)):
        for j_pos in range(i_pos + 1, len(indices)):
            all_pairs += 1
            s = mat[indices[i_pos]][indices[j_pos]]
            vals.append(s)
            if s > 0.0:
                pos += 1
    mean = sum(vals) / len(vals) if vals else 0.0
    connectivity = (float(pos) / float(all_pairs)) if all_pairs else 0.0
    return mean, connectivity


def _strongest_compat_partner(mat: List[List[float]], idx: int, n: int) -> Tuple[Optional[int], float]:
    best_j: Optional[int] = None
    best_s = -1.0
    for j in range(n):
        if j == idx:
            continue
        s = float(mat[idx][j])
        if best_j is None or s > best_s:
            best_s = s
            best_j = j
    if best_j is None:
        return None, 0.0
    return best_j, best_s


def _initial_augment_indices(
    seed_indices: Sequence[int],
    *,
    mat: List[List[float]],
    table_ids: Sequence[str],
    retrieval_score_map: Dict[str, float],
    threshold: float,
) -> List[int]:
    """
    Initial table augmentation: for each seed index, optionally add its strongest
    compatibility neighbor above threshold (mirrors table_selector seed augmentation).
    """
    n = len(mat)
    seed_list = list(dict.fromkeys(int(i) for i in seed_indices if 0 <= int(i) < n))
    if not seed_list:
        return []

    def raw_rel_score(table_index: int) -> float:
        if table_index < 0 or table_index >= len(table_ids):
            return float("-inf")
        key = _normalize_table_id(table_ids[table_index])
        val = retrieval_score_map.get(key)
        if val is None:
            val = retrieval_score_map.get(key.lower())
        return float(val) if isinstance(val, (int, float)) else float("-inf")

    augmented = list(seed_list)
    aug_set = set(augmented)
    for idx in seed_list:
        best_j, best_s = _strongest_compat_partner(mat, idx, n)
        if best_j is None or best_j == idx:
            continue
        if not (isinstance(best_s, (int, float)) and float(best_s) > float(threshold)):
            continue
        if 0 <= best_j < n and best_j not in aug_set:
            augmented.append(best_j)
            aug_set.add(best_j)

    selected_scores = [raw_rel_score(i) for i in seed_list]
    has_scores = bool(selected_scores) and any(s != float("-inf") for s in selected_scores)
    if not has_scores:
        return list(seed_list)
    if len(augmented) == len(seed_list):
        return list(seed_list)

    orig_set = set(seed_list)
    aug_only = [c for c in augmented if c not in orig_set]
    aug_sorted = sorted(aug_only, key=raw_rel_score, reverse=True)
    return list(seed_list) + list(aug_sorted)


def _indices_to_table_ids(table_ids: Sequence[str], indices: Sequence[int]) -> List[str]:
    out: List[str] = []
    for i in indices:
        if isinstance(i, int) and 0 <= i < len(table_ids):
            t = str(table_ids[i])
            if t:
                out.append(t)
    return out


def _greedy_seeded_expansion_from_indices(
    seed_indices: Sequence[int],
    *,
    n: int,
    mat: List[List[float]],
    rel_norm: Dict[int, float],
    max_size: int,
    edge_min: float,
    independent_candidate_gains: bool,
    add_all_above_min_gain: bool,
) -> List[int]:
    """Seeded expansion: greedily add tables from the top-K pool until max_size or no gain."""
    group = list(dict.fromkeys(int(i) for i in seed_indices if 0 <= int(i) < n))
    if not group:
        return []
    gain_baseline = list(group)
    max_size = max(1, int(max_size))
    edge_min = max(0.0, min(1.0, float(edge_min)))
    w_rel, w_coh, min_gain = 0.5, 0.5, 0.00
    candidates_idx = [i for i in range(n) if i not in set(group)]

    while len(group) < max_size:
        stats_group = gain_baseline if independent_candidate_gains else group
        cur_mean_pair, _ = _pair_stats(stats_group, mat)
        seed_mean_rel = (
            sum(rel_norm.get(j, 0.0) for j in stats_group) / len(stats_group) if stats_group else 0.0
        )
        best_i: Optional[int] = None
        best_gain = -1e9
        qualifying: List[Tuple[float, int]] = []
        for i in candidates_idx:
            if i in group:
                continue
            links = [mat[i][g] for g in stats_group]
            if not links:
                continue
            max_link = max(links)
            if max_link < edge_min:
                continue
            new_group = stats_group + [i]
            new_mean_pair, _ = _pair_stats(new_group, mat)
            cand_mean_rel = (
                sum(rel_norm.get(j, 0.0) for j in new_group) / len(new_group) if new_group else rel_norm.get(i, 0.0)
            )
            delta_avg_rel = cand_mean_rel - seed_mean_rel
            gain = (w_coh * (new_mean_pair - cur_mean_pair)) + (w_rel * delta_avg_rel)
            if gain > best_gain:
                best_gain = gain
                best_i = i
            if add_all_above_min_gain and gain > min_gain:
                qualifying.append((gain, i))
        if add_all_above_min_gain:
            if not qualifying:
                break
            qualifying.sort(key=lambda t: (-t[0], t[1]))
            added_any = False
            for _g, idx in qualifying:
                if len(group) >= max_size:
                    break
                if idx not in group:
                    group.append(idx)
                    added_any = True
            if not added_any:
                break
        else:
            if best_i is None or best_gain <= min_gain:
                break
            group.append(best_i)
    return list(group)


def _run_initial_augmentation_and_seeded_expansion(
    table_ids: Sequence[str],
    mat: List[List[float]],
    rel_norm: Dict[int, float],
    *,
    selection_indices: List[int],
    retrieval_score_map: Dict[str, float],
    initial_augment_threshold: float,
    expansion_max_size: int,
    expansion_edge_min: float,
    expansion_independent_candidate_gains: bool,
    expansion_add_all_above_min_gain: bool,
) -> Tuple[List[int], List[int]]:
    """
    One pipeline: map selected tables to indices → initial table augmentation → seeded expansion.

    Returns ``(indices_after_initial_augmentation, indices_after_seeded_expansion)``.
    """
    n = len(mat)
    raw = list(dict.fromkeys(int(i) for i in selection_indices if 0 <= int(i) < n))
    if not raw:
        return [], []
    after_initial = _initial_augment_indices(
        raw,
        mat=mat,
        table_ids=table_ids,
        retrieval_score_map=retrieval_score_map,
        threshold=float(initial_augment_threshold),
    )
    after_expansion = _greedy_seeded_expansion_from_indices(
        after_initial,
        n=n,
        mat=mat,
        rel_norm=rel_norm,
        max_size=expansion_max_size,
        edge_min=expansion_edge_min,
        independent_candidate_gains=expansion_independent_candidate_gains,
        add_all_above_min_gain=expansion_add_all_above_min_gain,
    )
    return after_initial, after_expansion


def _normalize_retrieval_scores(table_ids: Sequence[str], score_map: Dict[str, float]) -> Dict[int, float]:
    vals = []
    for t in table_ids:
        vals.append(float(score_map.get(_normalize_table_id(t), 0.0)))
    if not vals:
        return {}
    v_min = min(vals)
    v_max = max(vals)
    denom = (v_max - v_min) if v_max > v_min else 1.0
    out: Dict[int, float] = {}
    for idx, v in enumerate(vals):
        out[idx] = (v - v_min) / denom
    return out


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Augment table-selector output (initial augmentation + seeded expansion).")
    p.add_argument("--dataset", dest="dataset", choices=[dt.value for dt in DatabaseType], default=DatabaseType.BIRD.value)
    p.add_argument("--database-type", dest="dataset", choices=[dt.value for dt in DatabaseType], default=DatabaseType.BIRD.value)
    p.add_argument("--llm-model", type=str, default="openai:gpt-4o-mini")
    p.add_argument("--embedding-model", type=str, default="fireworks:WhereIsAI/UAE-Large-V1")
    p.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Dense-retriever K for input paths; default for --expansion-max-size when omitted.",
    )
    p.add_argument("--queries-file", type=str, default=None)
    p.add_argument("--data", type=str, default=None, help="Dense retrieval top-k JSON path.")
    p.add_argument("--retrieved-scores", type=str, default=None, help="Dense retrieval with-scores JSON path.")
    p.add_argument(
        "--selected-tables",
        type=str,
        default=None,
        dest="selected_tables_path",
        help="Path to selected_tables.json (default: results_table_selection/selected_tables.json for this run).",
    )
    p.add_argument("--similarities", type=str, default=None, help="Compatibility cache directory.")
    p.add_argument(
        "--expansion-max-size",
        type=int,
        default=None,
        help="Maximum tables after seeded expansion (default: --top-k).",
    )
    p.add_argument(
        "--expansion-edge-min",
        type=float,
        default=0.01,
        help="Minimum best edge compatibility to add a table during seeded expansion.",
    )
    p.add_argument(
        "--expansion-independent-candidate-gains",
        action="store_true",
        default=False,
        help="Score expansion candidates against the post-initial-augmentation group only.",
    )
    p.add_argument(
        "--expansion-add-all-above-min-gain",
        action="store_true",
        default=False,
        help="Each expansion step add every candidate above min gain (sorted), up to max size.",
    )
    p.add_argument(
        "--initial-augment-threshold",
        type=float,
        default=0.5,
        help="Compatibility threshold for initial table augmentation (matches table_selector --threshold).",
    )
    p.add_argument(
        "--initial-augment-out",
        type=str,
        default=None,
        help="Output path for augmented_tables_without_expansion.json (default: output directory).",
    )
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--evaluate-group",
        action="store_true",
        default=False,
        help="Evaluate augmented tables against gold and write eval_summary.json.",
    )
    p.add_argument(
        "--gold",
        type=str,
        default=None,
        help="Path to gold tables JSON (defaults to data/<dataset>/dev_gold_tables.json).",
    )
    p.add_argument("--eval-out", type=str, default=None, help="Optional path for eval_summary.json.")
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for outputs (default: results/<dataset>/{models}/results_table_augmentation/).",
    )
    return p.parse_args(argv)


def main(args: argparse.Namespace) -> int:
    run_dir = get_results_run_dir(
        dataset=str(args.dataset),
        step_dirname="results_dense_retriever",
        llm_model=str(args.llm_model),
        embedding_model=str(args.embedding_model),
        project_root=PROJECT_ROOT,
    )
    top_k = max(1, int(args.top_k))
    data_path = (
        Path(args.data).expanduser().resolve()
        if args.data
        else (run_dir / f"{args.dataset}_k_{int(top_k)}.json").resolve()
    )
    scores_path = (
        Path(args.retrieved_scores).expanduser().resolve()
        if args.retrieved_scores
        else (run_dir / f"{args.dataset}_k_{int(top_k)}_with_scores.json").resolve()
    )
    selection_run_dir = get_results_run_dir(
        dataset=str(args.dataset),
        step_dirname="results_table_selection",
        llm_model=str(args.llm_model),
        embedding_model=str(args.embedding_model),
        project_root=PROJECT_ROOT,
    )
    selected_tables_path = (
        Path(args.selected_tables_path).expanduser().resolve()
        if args.selected_tables_path
        else (selection_run_dir / "selected_tables.json").resolve()
    )
    compat_dir = Path(args.similarities).expanduser().resolve() if args.similarities else (PROJECT_ROOT / "cache" / str(args.dataset)).resolve()
    if not args.similarities:
        compat_dir = (
            PROJECT_ROOT
            / "cache"
            / str(args.dataset)
            / "".join(c if c.isalnum() else "_" for c in f"{args.llm_model}_{args.embedding_model}")
            / "compatibility"
        ).resolve()

    queries_path = (
        Path(args.queries_file).expanduser().resolve()
        if args.queries_file
        else (PROJECT_ROOT / "data" / str(args.dataset) / "dev.json").resolve()
    )
    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else get_results_run_dir(
            dataset=str(args.dataset),
            step_dirname="results_table_augmentation",
            llm_model=str(args.llm_model),
            embedding_model=str(args.embedding_model),
            project_root=PROJECT_ROOT,
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    augmented_tables_path = out_dir / "augmented_tables.json"
    augmented_tables_scores_path = out_dir / "augmented_tables_with_scores.json"
    augmented_without_expansion_path = out_dir / "augmented_tables_without_expansion.json"
    augmented_details_path = out_dir / "augmented_tables_details.json"

    selected_tables_payload: List[List[str]] = []
    if selected_tables_path.exists():
        selected_tables_payload = _load_selected_tables(selected_tables_path)
        print(f"Using selected tables from: {selected_tables_path}")
    else:
        print(f"⚠️  selected_tables.json not found: {selected_tables_path} (empty selection per query)")

    queries = _load_queries(queries_path)
    retrieved = _load_retrieved_tables(data_path)
    scores_payload = _load_retrieved_scores(scores_path if scores_path.exists() else None)

    n = min(len(queries), len(retrieved))
    if n == 0:
        print("No queries/retrieved tables to process.")
        return 1

    start = max(0, int(args.start))
    end = n if args.limit is None else min(n, start + int(args.limit))
    if start >= end:
        print("Nothing to process for this start/limit slice.")
        return 0

    expansion_max_size = int(args.expansion_max_size) if args.expansion_max_size is not None else top_k
    expansion_edge_min = float(args.expansion_edge_min)
    initial_augment_threshold = float(args.initial_augment_threshold)

    initial_augment_tables_out: List[List[str]] = []
    augmented_tables_out: List[List[str]] = []
    augmented_tables_with_scores_out: List[List[Dict[str, Any]]] = []
    details_rows: List[Dict[str, Any]] = []

    for idx in range(start, end):
        q_obj = queries[idx] if idx < len(queries) else {}
        query_text = str((q_obj or {}).get("question", "") or "")
        db_id = str((q_obj or {}).get("db_id", "") or "").strip().lower()

        aug_summary = {
            "dense_retrieval_top_k": int(top_k),
            "expansion_max_size": int(expansion_max_size),
            "expansion_edge_min": float(expansion_edge_min),
            "initial_augment_threshold": float(initial_augment_threshold),
        }

        table_ids = [_normalize_table_id(t) for t in (retrieved[idx] if idx < len(retrieved) else [])]
        table_ids = [t for t in table_ids if t]
        if not table_ids:
            initial_augment_tables_out.append([])
            augmented_tables_out.append([])
            augmented_tables_with_scores_out.append([])
            details_rows.append(
                {
                    "question_id": idx,
                    "db_id": db_id,
                    "query": query_text,
                    "retrieved_tables": [],
                    "llm_selected_tables": [],
                    "augmented_tables_only": [],
                    "augmented_tables_without_expansion": [],
                    "expansion_only_tables": [],
                    "selected_tables": [],
                    "augmentation_summary": aug_summary,
                }
            )
            continue

        retrieval_score_map: Dict[str, float] = {}
        if scores_payload is not None and idx < len(scores_payload):
            for entry in (scores_payload[idx] or []):
                try:
                    t = _normalize_table_id(entry.get("table"))
                    retrieval_score_map[t] = float(entry.get("similarity_score", 0.0))
                except Exception:
                    continue
        rel_norm = _normalize_retrieval_scores(table_ids, retrieval_score_map)
        compat_matrix = _build_compat_matrix(compat_dir, table_ids)

        selection_indices: List[int] = []
        llm_selected_tables: List[str] = []
        if 0 <= idx < len(selected_tables_payload):
            selected_seed_tables = selected_tables_payload[idx] or []
            table_pos = {t: i for i, t in enumerate(table_ids)}
            selection_indices = [
                table_pos[t] for t in selected_seed_tables if isinstance(t, str) and t in table_pos
            ]
            selection_indices = list(dict.fromkeys(selection_indices))
            llm_selected_tables = [_normalize_table_id(table_ids[i]) for i in selection_indices if 0 <= i < len(table_ids)]
            llm_selected_tables = [str(t) for t in llm_selected_tables if str(t)]

        after_initial: List[int] = []
        after_expansion: List[int] = []
        if selection_indices:
            after_initial, after_expansion = _run_initial_augmentation_and_seeded_expansion(
                table_ids,
                compat_matrix,
                rel_norm,
                selection_indices=selection_indices,
                retrieval_score_map=retrieval_score_map,
                initial_augment_threshold=initial_augment_threshold,
                expansion_max_size=expansion_max_size,
                expansion_edge_min=expansion_edge_min,
                expansion_independent_candidate_gains=bool(
                    getattr(args, "expansion_independent_candidate_gains", False)
                ),
                expansion_add_all_above_min_gain=bool(getattr(args, "expansion_add_all_above_min_gain", False)),
            )

        aug_flat = _indices_to_table_ids(table_ids, after_initial)
        initial_augment_tables_out.append(aug_flat)

        picked_indices = after_expansion
        final_tables = _indices_to_table_ids(table_ids, picked_indices)

        llm_set = set(llm_selected_tables)
        augmented_tables_only = [t for t in aug_flat if t not in llm_set]
        baseline_set = set(aug_flat)
        expansion_only_tables = [t for t in final_tables if t not in baseline_set] if baseline_set else []

        score_map_row: Dict[str, float] = {}
        if scores_payload is not None and 0 <= idx < len(scores_payload):
            for entry in (scores_payload[idx] or []):
                if not isinstance(entry, dict):
                    continue
                try:
                    tid = _normalize_table_id(entry.get("table"))
                    if tid:
                        score_map_row[tid] = float(entry.get("similarity_score", 0.0))
                except Exception:
                    continue

        augmented_tables_out.append(final_tables)
        augmented_tables_with_scores_out.append(
            [
                {
                    "table": str(t),
                    "similarity_score": float(score_map_row.get(str(t), 0.0)),
                    "rerank_score": None,
                }
                for t in final_tables
                if str(t)
            ]
        )
        details_rows.append(
            {
                "question_id": idx,
                "db_id": db_id,
                "query": query_text,
                "retrieved_tables": table_ids,
                "llm_selected_tables": llm_selected_tables,
                "augmented_tables_only": augmented_tables_only,
                "augmented_tables_without_expansion": list(aug_flat),
                "expansion_only_tables": expansion_only_tables,
                "selected_tables": final_tables,
                "augmentation_summary": aug_summary,
            }
        )

    _write_json(augmented_tables_path, augmented_tables_out)
    _write_json(augmented_tables_scores_path, augmented_tables_with_scores_out)
    _write_json(augmented_details_path, details_rows)
    print(f"Wrote augmented tables: {augmented_tables_path}")
    print(f"Wrote augmented tables with scores: {augmented_tables_scores_path}")
    print(f"Wrote augmented tables details: {augmented_details_path}")
    print(f"Dense retrieval top_k={top_k} | data: {data_path}")

    augment_path = (
        Path(args.initial_augment_out).expanduser().resolve()
        if getattr(args, "initial_augment_out", None)
        else augmented_without_expansion_path
    )
    _write_json(augment_path, initial_augment_tables_out)
    print(f"Wrote augmented tables without expansion (initial augmentation): {augment_path}")

    if bool(getattr(args, "evaluate_group", False)):
        gold_path = (
            Path(args.gold).expanduser().resolve()
            if args.gold
            else (PROJECT_ROOT / "data" / str(args.dataset) / "dev_gold_tables.json").resolve()
        )
        gold_tables = _load_gold_tables(gold_path)
        if not gold_tables:
            print(f"⚠️  Evaluation requested but no usable gold tables found at: {gold_path}")
            return 0

        aligned_preds: List[List[str]] = [[] for _ in range(len(gold_tables))]
        group_sizes: List[float] = []

        for row in details_rows:
            q_idx = int(row.get("question_id", -1))
            if q_idx < 0 or q_idx >= len(gold_tables):
                continue
            preds = row.get("selected_tables", []) if isinstance(row.get("selected_tables"), list) else []
            aligned_preds[q_idx] = [str(x) for x in preds]
            if preds:
                group_sizes.append(float(len(preds)))

        summary, _ = _evaluate(aligned_preds, gold_tables)
        stats = {
            "augmented_group_tables_per_query": {
                "min": _safe_min(group_sizes),
                "max": _safe_max(group_sizes),
                "avg": _safe_avg(group_sizes),
                "value_counts": _value_counts_full_range(group_sizes),
            },
            "totals": {"augmented_group_tables_all_queries": float(sum(group_sizes))},
        }
        eval_report = {
            "mode": "table_augmentor",
            "pred_path": str(augmented_tables_path),
            "gold_path": str(gold_path),
            "dense_retrieval_top_k": int(top_k),
            "stats": stats,
            **summary,
        }
        eval_out = Path(args.eval_out).expanduser().resolve() if args.eval_out else (out_dir / "eval_summary.json")
        _write_json(eval_out, eval_report)
        macro = eval_report.get("macro", {}) if isinstance(eval_report, dict) else {}
        print(
            "augmented group vs gold | "
            f"P={float(macro.get('precision', 0.0)):.3f} "
            f"R={float(macro.get('recall', 0.0)):.3f} "
            f"F1={float(macro.get('f1', 0.0)):.3f} "
            f"perfect_recall={float(macro.get('perfect_recall', 0.0)):.3f}"
        )
        print(f"Wrote evaluation summary: {eval_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(_parse_args()))
