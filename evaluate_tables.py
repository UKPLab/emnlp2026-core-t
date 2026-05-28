#!/usr/bin/env python3
"""
Evaluate table predictions against gold tables.

You can evaluate ONE file at a time:
  - dense retriever output  (--mode retrieved)
  - table selector output  (--mode selected)
  - augmented tables output  (--mode augmented)
  - custom input (--mode custom + --custom-data-path pointing to the prediction JSON file; eval outputs go next to it)

Usage:
    python evaluate_tables.py --dataset bird --mode retrieved --llm-model "huggingface:Qwen/Qwen2.5-7B-Instruct" --embedding-model "fireworks:WhereIsAI/UAE-Large-V1"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv


TABLE_KEY_SEP = "#sep#"


def _project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "data").exists():
            return parent
    return current.parent


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Best-effort env loading; evaluation should still work without it.
try:
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

from utils import get_results_run_dir  # noqa: E402


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=indent)


def _parse_partition_suffix(name: str) -> Optional[Tuple[int, int]]:
    """
    Parse a suffix of the form `_<partition>_of_<num_partitions>` from a filename stem.

    Example:
      selected_tables_0_of_2.json -> (0, 2)
    """
    stem = str(name or "")
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    try:
        # ... _ <partition> _ of _ <num_partitions>
        if parts[-2] != "of":
            return None
        partition = int(parts[-3])
        num_partitions = int(parts[-1])
        if num_partitions <= 0 or partition < 0 or partition >= num_partitions:
            return None
        return partition, num_partitions
    except Exception:
        return None


def _discover_partition_files(run_dir: Path, *, details: bool = False) -> List[Path]:
    """
    Discover partitioned table-selector outputs in `run_dir`.

    - selected tables: selected_tables_<p>_of_<n>.json
    - details:         selected_tables_details_<p>_of_<n>.json
    """
    if details:
        pattern = "selected_tables_details_*_of_*.json"
    else:
        pattern = "selected_tables_*_of_*.json"
    candidates = sorted(run_dir.glob(pattern))
    out: List[Path] = []
    for p in candidates:
        if not p.is_file():
            continue
        # Validate suffix parse (ignore unrelated files that happen to match the glob).
        if _parse_partition_suffix(p.stem) is None:
            continue
        out.append(p)
    return out


def _merge_partitioned_selected_tables(
    *,
    run_dir: Path,
    n_total: int,
    indent: int,
) -> Tuple[Path, Optional[Path]]:
    """
    Merge partitioned selected-tables outputs into `selected_tables.json` (and details into
    `selected_tables_details.json` when available).

    Prefers details-based merging (aligns by question_id). Falls back to concatenation
    of partitioned `selected_tables_<p>_of_<n>.json` in ascending p when details are absent.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    merged_tables_path = (run_dir / "selected_tables.json").resolve()
    merged_details_path = (run_dir / "selected_tables_details.json").resolve()

    part_details = _discover_partition_files(run_dir, details=True)
    part_tables = _discover_partition_files(run_dir, details=False)

    if not part_details and not part_tables:
        raise FileNotFoundError(f"No partitioned selected-tables outputs found under: {run_dir}")

    # 1) Best path: merge using details (stable alignment by question_id).
    if part_details:
        merged: List[List[str]] = [[] for _ in range(max(0, int(n_total)))]
        all_details: List[Dict[str, Any]] = []
        filled = 0
        for fp in part_details:
            payload = _read_json(fp)
            if not isinstance(payload, list):
                continue
            for entry in payload:
                if not isinstance(entry, dict):
                    continue
                idx_raw = entry.get("question_id", entry.get("index", None))
                try:
                    idx = int(idx_raw)
                except Exception:
                    continue
                if idx < 0 or idx >= len(merged):
                    continue
                pred_tables = entry.get("selected_tables")
                if isinstance(pred_tables, list):
                    merged[idx] = [_normalize_table_id(t) for t in pred_tables if _normalize_table_id(t)]
                    filled += 1
                all_details.append(entry)

        # Save merged artifacts for downstream pipeline steps.
        _write_json(merged_tables_path, merged, indent=int(indent))
        # Sort details for readability/determinism.
        try:
            all_details_sorted = sorted(
                [d for d in all_details if isinstance(d, dict)],
                key=lambda d: int(d.get("question_id", d.get("index", 0))),
            )
        except Exception:
            all_details_sorted = [d for d in all_details if isinstance(d, dict)]
        _write_json(merged_details_path, all_details_sorted, indent=int(indent))

        if filled == 0:
            # Details existed but no usable indices; keep details merged file, but warn by raising.
            raise RuntimeError(f"Found partitioned details but could not extract any selected_tables entries in {run_dir}")

        return merged_tables_path, merged_details_path

    # 2) Fallback: concatenate partitioned tables in partition order (only safe when partitions cover the full dataset in order).
    parts: List[Tuple[int, int, Path]] = []
    for fp in part_tables:
        parsed = _parse_partition_suffix(fp.stem)
        if parsed is None:
            continue
        p_idx, p_n = parsed
        parts.append((p_idx, p_n, fp))
    parts.sort(key=lambda t: t[0])

    merged_list: List[List[str]] = []
    for _p_idx, _p_n, fp in parts:
        merged_list.extend(_coerce_predictions(_read_json(fp)))

    _write_json(merged_tables_path, merged_list, indent=int(indent))
    return merged_tables_path, None


def _avg_table_count(preds: List[List[str]]) -> float:
    if not preds:
        return 0.0
    total = 0
    for entry in preds:
        if not isinstance(entry, list):
            continue
        uniq = {t for t in (_normalize_table_id(x) for x in entry) if t}
        total += len(uniq)
    return float(total) / float(len(preds))


def _usage_metrics_from_selection_details(details_payload: Any) -> Optional[Dict[str, Any]]:
    """
    Extract aggregate usage/error metrics from table_selector details payload.

    Expected shape: list[dict] where each dict has `cluster_summary.llm_usage` and `cluster_summary.llm_error`.
    """
    if not isinstance(details_payload, list) or not details_payload:
        return None

    total_in = 0.0
    total_out = 0.0
    total_total = 0.0
    total_cost = 0.0
    err_count = 0
    n = 0

    def _as_float(x: Any) -> float:
        try:
            if x is None:
                return 0.0
            return float(x)
        except Exception:
            return 0.0

    for entry in details_payload:
        if not isinstance(entry, dict):
            continue
        n += 1
        cs = entry.get("cluster_summary") if isinstance(entry.get("cluster_summary"), dict) else {}
        llm_error = bool(cs.get("llm_error", False)) if isinstance(cs, dict) else False
        err_count += 1 if llm_error else 0

        usage = cs.get("llm_usage") if isinstance(cs, dict) and isinstance(cs.get("llm_usage"), dict) else {}
        total_in += _as_float(usage.get("input_tokens") or usage.get("prompt_tokens"))
        total_out += _as_float(usage.get("output_tokens") or usage.get("completion_tokens"))
        total_total += _as_float(usage.get("total_tokens"))
        total_cost += _as_float(usage.get("cost_usd") or usage.get("cost") or usage.get("usd_cost"))

    if n == 0:
        return None

    return {
        "n_with_details": n,
        "totals": {
            "input_tokens": total_in,
            "output_tokens": total_out,
            "total_tokens": total_total,
            "cost_usd": total_cost,
        },
        "averages": {
            "input_tokens": total_in / n,
            "output_tokens": total_out / n,
            "total_tokens": total_total / n,
            "cost_usd": total_cost / n,
        },
        "llm_error_rate": float(err_count) / float(n),
    }


def _normalize_table_id(table_id: Any) -> str:
    s = str(table_id or "").strip()
    if not s:
        return ""
    if TABLE_KEY_SEP in s:
        dbp, tp = s.split(TABLE_KEY_SEP, 1)
        return f"{dbp.strip().lower()}{TABLE_KEY_SEP}{tp.strip().lower()}"
    return s.lower()


def _gold_to_full_ids(gold_entry: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    """
    Convert one gold entry to (db_id, question, full_table_ids).

    Gold format (BIRD):
      {"db_id": "...", "question": "...", "tables": ["t1","t2", ...], ...}
    """
    db_id = str(gold_entry.get("db_id", "") or "").strip().lower()
    question = str(gold_entry.get("question", "") or "")
    tables = gold_entry.get("tables", []) or []
    out: List[str] = []
    if isinstance(tables, list):
        for t in tables:
            tname = str(t or "").strip().lower()
            if not tname:
                continue
            out.append(_normalize_table_id(f"{db_id}{TABLE_KEY_SEP}{tname}") if db_id else tname)
    return db_id, question, out


def _load_gold(path: Path) -> Tuple[List[Dict[str, Any]], List[List[str]]]:
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Gold file is not a list: {path}")
    gold_entries: List[Dict[str, Any]] = [x for x in payload if isinstance(x, dict)]
    gold_tables_per_query: List[List[str]] = []
    for entry in gold_entries:
        _, _, full_ids = _gold_to_full_ids(entry)
        gold_tables_per_query.append(full_ids)
    return gold_entries, gold_tables_per_query


def _coerce_predictions(payload: Any) -> List[List[str]]:
    """
    Accepts:
      - list[list[str]]
      - list[list[{"table": "..."}]] (dense retriever with scores)
      - list[dict] where dict might contain a "selected_tables" or similar list (best-effort)
    Returns list[list[str]] normalized.
    """
    if not isinstance(payload, list):
        return []
    out: List[List[str]] = []
    for entry in payload:
        if isinstance(entry, list):
            pred_list: List[str] = []
            for item in entry:
                if isinstance(item, str):
                    pred_list.append(_normalize_table_id(item))
                elif isinstance(item, dict) and "table" in item:
                    pred_list.append(_normalize_table_id(item.get("table")))
                else:
                    continue
            out.append([t for t in pred_list if t])
            continue
        if isinstance(entry, dict):
            for key in ("selected_tables", "retrieved_tables", "augmented_tables", "tables"):
                if isinstance(entry.get(key), list):
                    pred_list = [_normalize_table_id(x) for x in entry.get(key, [])]
                    out.append([t for t in pred_list if t])
                    break
            else:
                out.append([])
            continue
        out.append([])
    return out


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _metrics_for_one(pred: Sequence[str], gold: Sequence[str]) -> Dict[str, Any]:
    pred_set = {t for t in (_normalize_table_id(x) for x in (pred or [])) if t}
    gold_set = {t for t in (_normalize_table_id(x) for x in (gold or [])) if t}
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
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if (micro_precision + micro_recall) else 0.0

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


def _infer_mode_from_prediction_path(pred_path: Path) -> str:
    """
    Guess artifact type from filename for metrics branches (partition merge, selection usage).

    Defaults to ``retrieved`` when the name is ambiguous.
    """
    stem_l = pred_path.stem.lower()
    name_l = pred_path.name.lower()
    if name_l == "selected_tables.json" or stem_l.startswith("selected_tables"):
        return "selected"
    if name_l == "augmented_tables.json" or stem_l.startswith("augmented_tables"):
        return "augmented"
    # Dense retriever convention: <dataset>_k_<int>.json
    parts = stem_l.split("_k_")
    if len(parts) == 2 and parts[1].isdigit():
        return "retrieved"
    return "retrieved"


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate predicted tables against gold tables (one input file at a time).")
    p.add_argument(
        "--dataset",
        dest="dataset",
        type=str,
        default="bird",
        help="Used for default gold path (data/<dataset>/dev_gold_tables.json).",
    )
    # Backward-compatible alias.
    p.add_argument(
        "--database-type",
        dest="dataset",
        type=str,
        default="bird",
        help="(deprecated) Same as --dataset.",
    )
    p.add_argument("--llm-model", type=str, default="openai:gpt-4o-mini", help="LLM tag used for results run folder naming.")
    p.add_argument(
        "--embedding-model",
        type=str,
        default="fireworks:WhereIsAI/UAE-Large-V1",
        help="Embedding model tag used for results run folder naming.",
    )
    p.add_argument(
        "--mode",
        type=str,
        choices=["retrieved", "selected", "augmented", "custom"],
        default="retrieved",
        help="Which prediction artifact to evaluate when no explicit --retrieved/--selected/--augmented path is provided. "
        "Use 'custom' with --custom-data-path set to the prediction JSON file (eval outputs are written next to it).",
    )
    p.add_argument(
        "--custom-data-path",
        type=str,
        default=None,
        help="Required with --mode custom: path to the prediction JSON file to evaluate. "
        "Default eval_summary.json / eval_details.json are written in the same directory.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Used only for default dense-retriever prediction path when mode=retrieved (default: 10).",
    )
    p.add_argument(
        "--gold",
        type=str,
        default=None,
        help="Path to gold tables JSON (defaults to data/<database-type>/dev_gold_tables.json).",
    )
    group = p.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--retrieved",
        type=str,
        default=None,
        help="Dense retriever output JSON (list[list[str]] or list[list[{table,...}]]).",
    )
    group.add_argument(
        "--selected",
        type=str,
        default=None,
        help="Table selector output JSON (list[list[str]]).",
    )
    group.add_argument(
        "--augmented",
        type=str,
        default=None,
        help="Table augmentor output JSON (list[list[str]]), e.g. augmented_tables.json.",
    )
    p.add_argument("--out", type=str, default=None, help="Optional path to write a JSON summary report.")
    p.add_argument("--out-details", type=str, default=None, help="Optional path to write per-query details (can be large).")
    p.add_argument(
        "--write-details",
        action="store_true",
        default=False,
        help="If set, write details to the default output folder even when --out-details is not provided.",
    )
    p.add_argument("--indent", type=int, default=2, help="JSON indentation for outputs.")
    return p.parse_args(argv)


def main(args: argparse.Namespace) -> int:
    dataset = str(args.dataset or "bird").strip()
    gold_path = Path(args.gold).expanduser().resolve() if args.gold else (PROJECT_ROOT / "data" / dataset / "dev_gold_tables.json").resolve()
    if not gold_path.exists():
        print(f"❌ Gold file not found: {gold_path}")
        return 1

    gold_entries, gold_tables = _load_gold(gold_path)
    print(f"Loaded gold for {len(gold_tables)} questions: {gold_path}")

    mode: str
    pred_path: Path
    cli_mode = str(args.mode or "retrieved")

    if cli_mode == "custom":
        if not args.custom_data_path:
            print("❌ --mode custom requires --custom-data-path (path to the prediction JSON file).")
            return 1
        pred_path = Path(args.custom_data_path).expanduser().resolve()
        if pred_path.is_dir():
            print(f"❌ --custom-data-path must be a JSON file, not a directory: {pred_path}")
            return 1
        if pred_path.suffix.lower() != ".json":
            print(f"❌ --custom-data-path must be a .json file: {pred_path}")
            return 1
        mode = _infer_mode_from_prediction_path(pred_path)
    elif args.retrieved:
        mode = "retrieved"
        pred_path = Path(args.retrieved).expanduser().resolve()
    elif args.selected:
        mode = "selected"
        pred_path = Path(args.selected).expanduser().resolve()
    elif args.augmented:
        mode = "augmented"
        pred_path = Path(args.augmented).expanduser().resolve()
    else:
        mode = str(args.mode or "retrieved")
        if mode == "retrieved":
            run_dir = get_results_run_dir(
                dataset=str(dataset),
                step_dirname="results_dense_retriever",
                llm_model=str(args.llm_model),
                embedding_model=str(args.embedding_model),
                project_root=PROJECT_ROOT,
            )
            pred_path = (run_dir / f"{dataset}_k_{int(args.top_k)}.json").resolve()
        elif mode == "augmented":
            run_dir = get_results_run_dir(
                dataset=str(dataset),
                step_dirname="results_table_augmentation",
                llm_model=str(args.llm_model),
                embedding_model=str(args.embedding_model),
                project_root=PROJECT_ROOT,
            )
            pred_path = (run_dir / "augmented_tables.json").resolve()
        else:
            run_dir = get_results_run_dir(
                dataset=str(dataset),
                step_dirname="results_table_selection",
                llm_model=str(args.llm_model),
                embedding_model=str(args.embedding_model),
                project_root=PROJECT_ROOT,
            )
            pred_path = (run_dir / "selected_tables.json").resolve()

    # For table-selection outputs: if the full file is missing but partition files exist, merge them.
    merged_details_path: Optional[Path] = None
    if mode == "selected" and not pred_path.exists():
        # Try to merge partitioned outputs located next to the expected/explicit file.
        run_dir = pred_path.parent
        try:
            pred_path, merged_details_path = _merge_partitioned_selected_tables(
                run_dir=run_dir,
                n_total=len(gold_tables),
                indent=int(args.indent),
            )
            print(f"ℹ️  Merged partitioned selected tables into: {pred_path}")
            if merged_details_path:
                print(f"ℹ️  Merged partitioned selection details into: {merged_details_path}")
        except Exception as e:
            print(f"❌ Prediction file not found: {pred_path}")
            print(f"❌ Also failed to merge partitioned selected tables under {run_dir}: {e}")
            return 1

    if not pred_path.exists():
        print(f"❌ Prediction file not found: {pred_path}")
        return 1

    preds = _coerce_predictions(_read_json(pred_path))
    summary, per_query = _evaluate(preds, gold_tables)

    macro = summary["macro"]
    print(
        f"{mode} vs gold | "
        f"P={macro['precision']:.3f} "
        f"R={macro['recall']:.3f} "
        f"F1={macro['f1']:.3f} "
        f"perfect_recall={macro['perfect_recall']:.3f}"
    )

    report: Dict[str, Any] = {
        "mode": mode,
        "pred_path": str(pred_path),
        "gold_path": str(gold_path),
        **summary,
    }
    if cli_mode == "custom":
        report["layout"] = "custom"
        report["custom_data_path"] = str(pred_path)

    # Extra high-signal summary metrics.
    report["avg_pred_tables"] = _avg_table_count(preds)
    report["avg_gold_tables"] = _avg_table_count(gold_tables)

    # For table selection runs, include usage + error metrics when we can find details.
    if mode == "selected":
        selection_details_path = merged_details_path
        if selection_details_path is None:
            # If we have a full selected_tables.json, look for a sibling details file.
            selection_details_path = pred_path.with_name("selected_tables_details.json")
            if not selection_details_path.exists():
                # If only partitioned details exist but we didn't need to merge tables, aggregate them directly.
                part_details = _discover_partition_files(pred_path.parent, details=True)
                if part_details:
                    combined: List[Dict[str, Any]] = []
                    for fp in part_details:
                        payload = _read_json(fp)
                        if isinstance(payload, list):
                            combined.extend([x for x in payload if isinstance(x, dict)])
                    selection_details_path = None
                    usage_metrics = _usage_metrics_from_selection_details(combined)
                    if usage_metrics is not None:
                        report["selection_usage"] = usage_metrics
                else:
                    selection_details_path = None
        if selection_details_path is not None and selection_details_path.exists():
            usage_metrics = _usage_metrics_from_selection_details(_read_json(selection_details_path))
            if usage_metrics is not None:
                report["selection_usage"] = usage_metrics

    details: Optional[Dict[str, Any]] = None
    if args.out_details or bool(args.write_details):
        mini_gold: List[Dict[str, Any]] = []
        for entry in gold_entries:
            if not isinstance(entry, dict):
                continue
            db_id = str(entry.get("db_id", "") or "").strip().lower()
            question = str(entry.get("question", "") or "")
            tables = entry.get("tables", []) or []
            mini_gold.append(
                {
                    "db_id": db_id,
                    "question": question,
                    "tables": [str(t).strip().lower() for t in tables] if isinstance(tables, list) else [],
                }
            )

        details = {
            "mode": mode,
            "pred_path": str(pred_path),
            "gold_path": str(gold_path),
            "per_query": per_query,
            "gold": mini_gold,
        }

    # Default output folder:
    # - If caller provided --out, use that path.
    # - If --mode custom, write next to the custom prediction JSON.
    # - If caller provided --retrieved/--selected/--augmented (non-custom), write next to that file (same folder).
    # - Otherwise, write under the computed run folder.
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    elif cli_mode == "custom":
        out_path = (pred_path.parent / "eval_summary.json").resolve()
    elif args.retrieved or args.selected or args.augmented:
        out_path = (pred_path.parent / "eval_summary.json").resolve()
    else:
        if mode == "retrieved":
            step_dirname = "results_dense_retriever"
        elif mode == "augmented":
            step_dirname = "results_table_augmentation"
        else:
            step_dirname = "results_table_selection"
        out_path = (
            get_results_run_dir(
                dataset=str(dataset),
                step_dirname=step_dirname,
                llm_model=str(args.llm_model),
                embedding_model=str(args.embedding_model),
                project_root=PROJECT_ROOT,
            )
            / "eval_summary.json"
        ).resolve()
    _write_json(out_path, report, indent=int(args.indent))
    print(f"Wrote summary: {out_path}")

    if args.out_details or bool(args.write_details):
        out_details_path = (
            Path(args.out_details).expanduser().resolve()
            if args.out_details
            else out_path.with_name("eval_details.json")
        )
        _write_json(out_details_path, details or {}, indent=int(args.indent))
        print(f"Wrote details: {out_details_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(_parse_args()))

