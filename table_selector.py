#!/usr/bin/env python3
"""
Table selector.

This script reads dense-retriever outputs (top-K tables per query), then uses an
LLM to choose a coherent subset of tables per query.

Usage:
    python table_selector.py --dataset bird --llm-model "huggingface:Qwen/Qwen2.5-7B-Instruct" --embedding-model "fireworks:WhereIsAI/UAE-Large-V1"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

from utils import (
    Configuration,
    DatabaseType,
    get_results_run_dir,
    llm_usage_from_prompt_and_output,
    summarize_llm_invoke_error,
)
from utils.cache_scripts.selection_cache import get_selection_cache
from utils.prompts import get_selection_prompt_lrm_few_shot


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=False)


def _content_to_text(content: Any) -> str:
    """Normalize LangChain message content (string or structured parts) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # Some providers return a list of parts with `.text` or `.content`
    try:
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, str):
                    parts.append(p)
                elif isinstance(p, dict) and "text" in p:
                    parts.append(str(p.get("text", "")))
                else:
                    parts.append(str(getattr(p, "text", getattr(p, "content", p))))
            return "\n".join(parts)
    except Exception:
        pass
    return str(getattr(content, "content", content))


def _update_similarities_with_enriched_tables(
    similarities: List[Dict[str, Any]],
    enriched_tables: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    updated: List[Dict[str, Any]] = []
    for sim_pair in similarities or []:
        try:
            t1 = int(sim_pair["table1_index"])
            t2 = int(sim_pair["table2_index"])
        except Exception:
            continue
        if t1 < len(enriched_tables) and t2 < len(enriched_tables):
            new_pair = dict(sim_pair)
            new_pair["table1"] = enriched_tables[t1]
            new_pair["table2"] = enriched_tables[t2]
            updated.append(new_pair)
        else:
            print(f"Warning: invalid indices {t1}, {t2} for {len(enriched_tables)} tables")
    return updated


def _header_only_markdown(table_markdown: str, fallback_table_name: str = "") -> str:
    """Return table markdown limited to name and header (no sample rows)."""
    try:
        lines = str(table_markdown or "").splitlines()
        table_name_line = next((ln for ln in lines if ln.startswith("Table name: ")), None)
        if not table_name_line:
            table_name_line = f"Table name: {fallback_table_name}" if fallback_table_name else "Table name: "
        example_line = "Example table content:"
        header_idx = None
        separator_line = None
        for idx, ln in enumerate(lines):
            if ln.startswith("| "):
                header_idx = idx
                if idx + 1 < len(lines):
                    next_ln = lines[idx + 1]
                    stripped = next_ln.replace("|", "").replace(" ", "")
                    if stripped and all(ch in "-:" for ch in stripped):
                        separator_line = next_ln
                break
        if header_idx is None:
            return "\n".join([table_name_line, example_line]).strip()
        header_line = lines[header_idx]
        if separator_line is None:
            try:
                columns = [c.strip() for c in header_line.strip().strip("|").split("|")]
                separator_line = "|" + "|".join(["-------:" for _ in columns]) + "|"
            except Exception:
                separator_line = "|-------:|"
        return "\n".join([table_name_line, "Table header:", header_line]).strip()
    except Exception:
        return str(table_markdown or "")


def _build_tables_content(
    enriched_tables: List[Dict[str, Any]],
    similarities: Optional[List[Dict[str, Any]]] = None,
    *,
    headers_only: bool = False,
    include_purpose: bool = True,
    filter_by_similarity: bool = False,
    threshold: float = 0.5,
    retrieved_tables_scores: Optional[Dict[str, float]] = None,
) -> str:
    included_indices: Optional[set[int]] = None
    if filter_by_similarity and similarities:
        indices: set[int] = set()
        for sim_pair in similarities:
            try:
                overall = float(sim_pair.get("similarity", 0.0))
                if overall and overall > threshold:
                    indices.add(int(sim_pair.get("table1_index")))
                    indices.add(int(sim_pair.get("table2_index")))
            except Exception:
                continue
        included_indices = indices

    parts: List[str] = []
    for i, table in enumerate(enriched_tables):
        if table is None:
            continue
        if included_indices is not None and i not in included_indices:
            continue
        table_content = table.get("original_table", "")
        if headers_only:
            table_content = _header_only_markdown(table_content, table.get("table_name", ""))
        purpose = table.get("purpose", "No purpose available")

        score_suffix = ""
        try:
            if retrieved_tables_scores:
                db_id = str(table.get("db_id", "")).strip()
                table_name = str(table.get("table_name", "")).strip()
                key = f"{db_id}#sep#{table_name.lower()}" if db_id and table_name else table_name.lower()
                score_val = retrieved_tables_scores.get(key)
                if score_val is None:
                    score_val = retrieved_tables_scores.get(key.lower())
                if isinstance(score_val, (int, float)):
                    score_suffix = f" (Retrieval similarity={float(score_val):.3f})"
        except Exception:
            score_suffix = ""

        parts.append(
            "\n".join(
                [
                    f"Table {i}{score_suffix}:",
                    f"{table_content}",
                    f"Purpose: {purpose}" if include_purpose else "",
                    "",
                ]
            )
        )
    return "\n".join(parts)


def _build_compatibility_analysis(
    updated_similarities: List[Dict[str, Any]],
    *,
    include_best_join_columns: bool = False,
    filter_by_similarity: bool = False,
    threshold: float = 0.5,
) -> str:
    lines: List[str] = []
    for sim_pair in updated_similarities:
        try:
            i = int(sim_pair["table1_index"])
            j = int(sim_pair["table2_index"])
        except Exception:
            continue
        overall = sim_pair.get("similarity", 0.0)
        try:
            overall_f = float(overall)
        except Exception:
            overall_f = 0.0
        if overall_f == 0.0:
            continue
        if filter_by_similarity and not (overall_f and overall_f > threshold):
            continue
        details = sim_pair.get("compatibility_details", {}) or {}
        best = details.get("best_join_columns", ("", ""))
        if not best or not isinstance(best, (list, tuple)) or len(best) != 2:
            best = ("No clear join column", "No clear join column")
        table1 = sim_pair.get("table1", {}) or {}
        table2 = sim_pair.get("table2", {}) or {}
        table1_name = table1.get("table_name", "")
        table2_name = table2.get("table_name", "")
        lines.append(f"Pair (Table {i}: {table1_name} <-> Table {j}: {table2_name}):")
        if include_best_join_columns:
            try:
                lines.append(f"  Overall Compatibility: {overall_f:.3f}")
            except Exception:
                lines.append("  Overall Compatibility: 0.000")
        else:
            lines.append(f"  Overall Compatibility: {overall_f:.3f}")
        lines.append(f"  Best Join Columns: {best[0]} ↔ {best[1]}")
    return "\n".join(lines)


def _find_last_json_object_slice(text: str) -> Optional[str]:
    if not text:
        return None
    depth = 0
    in_string = False
    prev_escape = False
    start_idx: Optional[int] = None
    candidates: List[Tuple[int, int]] = []
    for idx, ch in enumerate(text):
        if in_string:
            if prev_escape:
                prev_escape = False
                continue
            if ch == "\\":
                prev_escape = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            prev_escape = False
            continue
        if ch == "{":
            if depth == 0:
                start_idx = idx
            depth += 1
            continue
        if ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start_idx is not None:
                    candidates.append((start_idx, idx + 1))
                    start_idx = None
    if not candidates:
        return None
    s, e = candidates[-1]
    return text[s:e]


def _parse_json_tolerant(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    # optional: json_repair
    try:
        from json_repair import repair_json

        repaired = repair_json(text)
        parsed = json.loads(repaired)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    return None


def _convert_group_to_clusters(
    enriched_tables: List[Dict[str, Any]],
    table_indices: List[int],
    *,
    group_index: int = 0,
    reasoning: str = "",
) -> List[Dict[str, Any]]:
    if not table_indices:
        return []
    cluster_tables = [enriched_tables[idx] for idx in table_indices if idx < len(enriched_tables)]
    cluster = {
        "cluster_id": f"llm_selected_group_{group_index}",
        "reasoning": reasoning,
        "table_indices": list(table_indices),
        "tables": cluster_tables,
        "size": len(table_indices),
    }
    return [cluster]


def _invoke_llm(model, prompt: str):
    """Invoke a LangChain chat model using various message formatting."""

    def _format_messages_for_model(model_name: str, system_prompt: str):
        from langchain_core.messages import SystemMessage, HumanMessage

        # OpenRouter: single human turn avoids fragile system-only routing for some upstreams.
        if getattr(model, "__class__", type(None)).__name__ == "ChatOpenRouter":
            return [HumanMessage(content=system_prompt)]

        name = str(model_name or "")
        lower = name.lower()

        # Ollama: no system message support in many setups; send as Human.
        if name.startswith("ollama:"):
            return [HumanMessage(content=system_prompt)]

        # Gemini / DeepInfra: some models need a trailing human turn to trigger.
        if name.startswith("google_genai:") or name.startswith("deepinfra:"):
            if "gemma" in lower or "gemini" in lower:
                return [HumanMessage(content=system_prompt)]
            return [SystemMessage(content=system_prompt), HumanMessage(content="Proceed.")]

        # Hugging Face chat wrappers: keep last message human.
        if name.startswith("huggingface:") or name.startswith("hf:"):
            return [SystemMessage(content=system_prompt), HumanMessage(content="Proceed.")]

        # Default: system-only.
        return [SystemMessage(content=system_prompt)]

    model_name = ""
    try:
        model_name = str(getattr(model, "model_name", "") or "")
    except Exception:
        model_name = ""

    # Prefer formatting (SystemMessage-first).
    try:
        messages = _format_messages_for_model(getattr(model, "model", model_name) or model_name, prompt)
        return model.invoke(messages)
    except Exception:
        # Fall back to simple invocation forms.
        try:
            return model.invoke(prompt)
        except Exception:
            try:
                from langchain_core.messages import HumanMessage

                return model.invoke([HumanMessage(content=prompt)])
            except Exception as e:
                raise RuntimeError(f"Failed to invoke LLM: {summarize_llm_invoke_error(e)}") from e


def _augment_selected_group(
    *,
    selected_group: List[int],
    selected_enriched_tables: List[Optional[Dict[str, Any]]],
    selected_similarities: List[Dict[str, Any]],
    retrieved_tables_scores: Optional[Dict[str, float]],
    threshold: float,
) -> List[int]:
    """
    Augment the selected tables with a small additive step.
    """

    def rel_score(idx: int) -> float:
        try:
            if not retrieved_tables_scores:
                return float("-inf")
            tbl = selected_enriched_tables[idx]
            if tbl is None:
                return float("-inf")
            db_id = str(tbl.get("db_id", "")).strip()
            table_name = str(tbl.get("table_name", "")).strip()
            key = f"{db_id}#sep#{table_name.lower()}" if db_id and table_name else table_name.lower()
            val = retrieved_tables_scores.get(key)
            if val is None:
                val = retrieved_tables_scores.get(key.lower())
            return float(val) if isinstance(val, (int, float)) else float("-inf")
        except Exception:
            return float("-inf")

    augmented_list: List[int] = list(selected_group)
    augmented_set = set(augmented_list)

    best_match_for: Dict[int, Tuple[int, float]] = {}
    for sp in selected_similarities or []:
        try:
            i = int(sp.get("table1_index"))
            j = int(sp.get("table2_index"))
            s = float(sp.get("similarity", 0.0))
        except Exception:
            continue
        if i not in best_match_for or s > best_match_for[i][1]:
            best_match_for[i] = (j, s)
        if j not in best_match_for or s > best_match_for[j][1]:
            best_match_for[j] = (i, s)

    for idx in selected_group:
        best_pair = best_match_for.get(idx)
        if not best_pair:
            continue
        best_idx, best_score = best_pair
        if best_idx is None or best_idx == idx:
            continue
        if not (isinstance(best_score, (int, float)) and best_score > threshold):
            continue
        if 0 <= best_idx < len(selected_enriched_tables):
            if selected_enriched_tables[best_idx] is not None and best_idx not in augmented_set:
                augmented_list.append(best_idx)
                augmented_set.add(best_idx)

    # Preserve existing behavior: if we don't have usable relevance scores, do not apply augmentation.
    selected_scores = [rel_score(i) for i in selected_group]
    has_scores = bool(selected_scores) and any(s != float("-inf") for s in selected_scores)
    if not has_scores:
        return list(selected_group)

    # No additions found.
    if len(augmented_list) == len(selected_group):
        return list(selected_group)

    # Keep the original selected order first; sort only the additions by relevance.
    try:
        aug_only = [c for c in augmented_list if c not in set(selected_group)]
        aug_sorted = sorted(aug_only, key=rel_score, reverse=True)
    except Exception:
        aug_sorted = [c for c in augmented_list if c not in set(selected_group)]
    return list(selected_group) + list(aug_sorted)


def _normalize_group_index(raw: Any, num_groups: int) -> int:
    """Map LLM/cache `selected_group_index` to a valid index in ``[0, num_groups)``.

    Models sometimes emit a list (e.g. ``[0]``) or a numeric string; treat those as
    a single index. Out-of-range or unparsable values fall back to ``0``.
    """
    if num_groups <= 0:
        return 0

    def _one(v: Any) -> Optional[int]:
        if v is None or isinstance(v, bool):
            return None
        if isinstance(v, int):
            return v if 0 <= v < num_groups else None
        if isinstance(v, float) and v == int(v) and v == v:  # not NaN
            iv = int(v)
            return iv if 0 <= iv < num_groups else None
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            try:
                iv = int(s)
            except ValueError:
                return None
            return iv if 0 <= iv < num_groups else None
        return None

    if isinstance(raw, list):
        for item in raw:
            got = _one(item)
            if got is not None:
                return got
        return 0

    got = _one(raw)
    return 0 if got is None else got


def _select_tables(
    enriched_tables: List[Dict[str, Any]],
    updated_similarities: List[Dict[str, Any]],
    query: str,
    model,
    selection_cache,
    *,
    config: Configuration,
    retrieved_tables_scores: Optional[Dict[str, float]] = None,
    augment_with_compatible_tables: bool = False,
    threshold: float = 0.12,
    use_cache: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]], Any]:
    # Work on local copies; do not mutate inputs.
    selected_enriched_tables: List[Optional[Dict[str, Any]]] = list(enriched_tables or [])
    selected_similarities: List[Dict[str, Any]] = list(updated_similarities or [])

    # Cache lookup (if enabled)
    if use_cache:
        cached = selection_cache.get_selection(selected_enriched_tables, query, config)
    else:
        cached = None

    if cached and isinstance(cached, dict):
        print(f"✅ Found selection cache for {len(selected_enriched_tables)} tables and query")
        selection_data = cached.get("selection_data", {}) or {}
        cached_clusters = selection_data.get("clusters", []) or []
        cluster_summary_cached = selection_data.get("cluster_summary", {}) or {}
        table_groups_cached = selection_data.get("table_groups", []) or []
        group_selection_cached = selection_data.get("group_selection", {}) or {}

        if augment_with_compatible_tables:
            try:
                raw_cached_idx = group_selection_cached.get("selected_group_index", -1)
                selection_reasoning = group_selection_cached.get(
                    "reasoning", cluster_summary_cached.get("selection_reasoning", "No group selection reasoning provided")
                )
                fallback_used = False
                if not table_groups_cached:
                    table_groups_cached = [list(range(len(enriched_tables)))]
                    selection_reasoning = "Cache missing groups; using all tables as fallback group"
                    fallback_used = True

                selected_group_index = _normalize_group_index(
                    raw_cached_idx, len(table_groups_cached)
                )
                if raw_cached_idx != selected_group_index and not isinstance(
                    raw_cached_idx, (int, type(None))
                ):
                    print(
                        f"⚠️  Coerced cached group index {raw_cached_idx!r} → {selected_group_index}"
                    )

                selected_group = table_groups_cached[selected_group_index] if table_groups_cached else []
                llm_selected_indices = list(selected_group) if isinstance(selected_group, list) else []
                augmented_only_indices: List[int] = []
                clusters = []
                if selected_group:
                    clusters = _convert_group_to_clusters(
                        selected_enriched_tables,
                        selected_group,
                        group_index=selected_group_index,
                        reasoning=selection_reasoning,
                    )

                    # Apply augmentation
                    augmented_group = _augment_selected_group(
                        selected_group=selected_group,
                        selected_enriched_tables=selected_enriched_tables,
                        selected_similarities=selected_similarities,
                        retrieved_tables_scores=retrieved_tables_scores,
                        threshold=threshold,
                    )
                    if augmented_group != selected_group:
                        try:
                            augmented_only_indices = sorted(list(set(augmented_group) - set(selected_group)))
                        except Exception:
                            augmented_only_indices = []
                        clusters = _convert_group_to_clusters(
                            selected_enriched_tables,
                            augmented_group,
                            group_index=selected_group_index,
                            reasoning=selection_reasoning + "; augmented",
                        )
                        print(f"   Augmentation added {len(augmented_group) - len(selected_group)} table(s) → new size {len(augmented_group)}")

                cluster_summary = dict(cluster_summary_cached or {})
                cluster_summary["total_clusters"] = len(clusters)
                cluster_summary["llm_error"] = bool(fallback_used) or bool(cluster_summary.get("llm_error", False))
                # Helpful for downstream logging: expose pre-augmentation and augmentation-only indices.
                cluster_summary["llm_selected_table_indices"] = llm_selected_indices
                cluster_summary["augmented_only_table_indices"] = augmented_only_indices
                print(f"Found {len(clusters)} clusters after cache-based augmentation")
                return clusters, cluster_summary, enriched_tables, model
            except Exception as e:
                print(f"⚠️  Failed to reuse cached selection for augmentation: {e}; returning cached clusters")
                return cached_clusters, cluster_summary_cached, selected_enriched_tables, model

        print(f"Found {len(cached_clusters)} clusters from cache")
        return cached_clusters, cluster_summary_cached, selected_enriched_tables, model

    # No cached result: ask the model to group/select tables.
    print(f"🧠 Selecting tables with the LLM for {len(selected_enriched_tables)} tables...")
    if model is None:
        try:
            model = config.create_llm()
        except ImportError as e:
            msg = str(e)
            # Common case: provider extras not installed (e.g., `together:` → langchain-together)
            print("\n❌ Unable to create LLM client for the requested model.")
            print(f"   Model: {getattr(config, 'llm_model', '')}")
            print(f"   Error: {msg}")
            if "langchain_together" in msg or "langchain-together" in msg:
                print("\nInstall the missing provider package:")
                print("  pip install -U langchain-together")
            raise

    tables_content = _build_tables_content(
        selected_enriched_tables,
        selected_similarities,
        headers_only=False,
        include_purpose=True,
        filter_by_similarity=False,
        threshold=threshold,
        retrieved_tables_scores=retrieved_tables_scores,
    )
    compatibility_analysis = _build_compatibility_analysis(
        selected_similarities,
        include_best_join_columns=False,
        filter_by_similarity=False,
        threshold=threshold,
    )

    prompt_content = get_selection_prompt_lrm_few_shot(query, tables_content, compatibility_analysis)

    # Invoke model, parse JSON
    cost_usd = 0.0
    model_response: Optional[Dict[str, Any]] = None
    llm_usage: Dict[str, Any] = {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "cost_usd": cost_usd,
    }

    try:
        ai_message = _invoke_llm(model, prompt_content)
        raw_content_str = _content_to_text(getattr(ai_message, "content", ""))
        llm_usage = llm_usage_from_prompt_and_output(
            model,
            prompt_content,
            raw_content_str,
            ai_message=ai_message,
            cost_usd=cost_usd,
        )

        # Parse JSON strictly first
        try:
            from langchain_core.output_parsers import JsonOutputParser

            json_parser = JsonOutputParser()
            model_response = json_parser.parse(raw_content_str)
        except Exception:
            # Fallback: extract last JSON object and parse tolerantly
            json_slice = _find_last_json_object_slice(raw_content_str)
            if json_slice:
                model_response = _parse_json_tolerant(json_slice)
            if model_response is None:
                model_response = _parse_json_tolerant(raw_content_str)
    except Exception as e:
        print(f"⚠️  LLM call failed: {e}")
        model_response = None

    if not model_response:
        print("⚠️  Invalid/empty model response. Falling back to using all tables.")
        table_groups = [list(range(len(enriched_tables)))] if enriched_tables else []
        selected_group_index = 0 if table_groups else -1
        llm_selected_indices = list(table_groups[0]) if table_groups else []
        clusters = _convert_group_to_clusters(
            selected_enriched_tables,
            table_groups[0] if table_groups else [],
            group_index=0,
            reasoning="Model selection failed; using all tables as fallback group",
        )
        cluster_summary = {
            "total_clusters": len(clusters),
            "selected_group_index": selected_group_index,
            "selection_reasoning": "Model selection failed; using all tables as fallback group",
            "total_groups_considered": len(table_groups),
            "group_analysis": [],
            "llm_usage": {**llm_usage, "cost_usd": cost_usd},
            "llm_error": True,
            "llm_selected_table_indices": llm_selected_indices,
            "augmented_only_table_indices": [],
        }
        if use_cache:
            try:
                selection_cache.set_selection(
                    enriched_tables,
                    query,
                    {
                        "clusters": clusters,
                        "cluster_summary": cluster_summary,
                        "table_groups": table_groups,
                        "group_selection": {
                            "selected_group_index": selected_group_index,
                            "reasoning": cluster_summary.get("selection_reasoning", ""),
                            "confidence": 0.0,
                        },
                    },
                    config,
                )
            except Exception:
                pass
        return clusters, cluster_summary, enriched_tables, model

    # Process model response
    group_formation = model_response.get("group_formation", {}) or {}
    groups_formed = group_formation.get("groups_formed", []) or []
    table_groups: List[List[int]] = []
    for group_data in groups_formed:
        table_indices = group_data.get("table_indices", []) or []
        valid_indices = [idx for idx in table_indices if isinstance(idx, int) and 0 <= idx < len(enriched_tables)]
        if valid_indices:
            table_groups.append(valid_indices)

    fallback_used = False
    if not table_groups:
        print("⚠️  No valid groups from LLM response, falling back to using all tables as one group")
        table_groups = [list(range(len(enriched_tables)))]
        fallback_used = True

    group_selection = model_response.get("group_selection", {}) or {}
    raw_selected_idx = group_selection.get("selected_group_index", -1)
    selection_reasoning = group_selection.get("reasoning", "No group selection reasoning provided")
    if fallback_used:
        selection_reasoning = "LLM group formation failed, using all tables as fallback group"

    selected_group_index = (
        0
        if fallback_used
        else _normalize_group_index(raw_selected_idx, len(table_groups))
    )
    if (
        not fallback_used
        and raw_selected_idx != selected_group_index
        and not isinstance(raw_selected_idx, (int, type(None)))
    ):
        print(f"⚠️  Coerced selected_group_index {raw_selected_idx!r} → {selected_group_index}")

    clusters: List[Dict[str, Any]] = []
    selected_group = table_groups[selected_group_index] if table_groups else []
    llm_selected_indices = list(selected_group) if isinstance(selected_group, list) else []
    augmented_only_indices: List[int] = []
    if selected_group:
        clusters = _convert_group_to_clusters(
            selected_enriched_tables,
            selected_group,
            group_index=int(selected_group_index),
            reasoning=str(selection_reasoning),
        )

    cluster_summary_pre = {
        "total_clusters": len(clusters),
        "selected_group_index": selected_group_index,
        "selection_reasoning": selection_reasoning,
        "total_groups_considered": len(table_groups),
        "group_analysis": group_selection.get("group_analysis") or model_response.get("group_analysis", []),
        "llm_usage": {**llm_usage, "cost_usd": cost_usd},
        "llm_error": bool(fallback_used),
        "llm_selected_table_indices": llm_selected_indices,
        "augmented_only_table_indices": augmented_only_indices,
    }

    if use_cache:
        try:
            selection_cache.set_selection(
                selected_enriched_tables,
                query,
                {
                    "clusters": clusters,
                    "cluster_summary": cluster_summary_pre,
                    "table_groups": table_groups,
                    "group_selection": group_selection,
                },
                config,
            )
        except Exception:
            pass

    if augment_with_compatible_tables and selected_group:
        try:
            augmented_group = _augment_selected_group(
                selected_group=selected_group,
                selected_enriched_tables=selected_enriched_tables,
                selected_similarities=selected_similarities,
                retrieved_tables_scores=retrieved_tables_scores,
                threshold=threshold,
            )
            if augmented_group != selected_group:
                try:
                    augmented_only_indices = sorted(list(set(augmented_group) - set(selected_group)))
                except Exception:
                    augmented_only_indices = []
                clusters = _convert_group_to_clusters(
                    selected_enriched_tables,
                    augmented_group,
                    group_index=int(selected_group_index),
                    reasoning=str(selection_reasoning) + "; augmented",
                )
                cluster_summary_pre = dict(cluster_summary_pre)
                cluster_summary_pre["total_clusters"] = len(clusters)
                cluster_summary_pre["augmented_only_table_indices"] = augmented_only_indices
        except Exception as aug_e:
            print(f"⚠️  Augmentation step failed: {aug_e}")

    return clusters, cluster_summary_pre, enriched_tables, model


def validate_and_cluster_tables(
    enriched_tables: List[Dict[str, Any]],
    similarities: List[Dict[str, Any]],
    query: str,
    *,
    config: Configuration,
    retrieved_tables_scores: Optional[Dict[str, float]] = None,
    augment_with_compatible_tables: bool = False,
    threshold: float = 0.12,
    use_cache: bool = True,
    model: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]], Any]:
    if len(enriched_tables) < 2 or not similarities:
        print("Less than 2 tables or no similarities, skipping clustering")
        return [], {}, enriched_tables, model

    print(f"Clustering {len(enriched_tables)} tables using {len(similarities)} similarity pairs...")
    updated_similarities = _update_similarities_with_enriched_tables(similarities, enriched_tables)
    print(f"Updated {len(updated_similarities)} similarity pairs with enriched data")

    cache_dir = config.get_database_cache_dir("selections")
    selection_cache = get_selection_cache(cache_dir)
    return _select_tables(
        enriched_tables,
        updated_similarities,
        query,
        model,
        selection_cache,
        config=config,
        retrieved_tables_scores=retrieved_tables_scores,
        augment_with_compatible_tables=augment_with_compatible_tables,
        threshold=threshold,
        use_cache=use_cache,
    )


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Standalone table selection over dense-retriever outputs. "
            "Loads per-query retrieved tables, pulls enriched-table and compatibility caches, "
            "then runs LLM grouping/selection."
        )
    )

    p.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to dense-retriever output JSON (defaults to results/<dataset>/{llm}_{embedding_model}/results_dense_retriever/<dataset>_k_<top_k>.json).",
    )
    p.add_argument(
        "--queries-file",
        type=str,
        default=None,
        help="Path to queries JSON (defaults to data/<dataset>/dev.json).",
    )
    p.add_argument(
        "--enriched-tables",
        type=str,
        required=False,
        default=None,
        help="Path to enriched tables cache directory (defaults to cache/<dataset>/{llm}_{embedding_model}/metadata/enriched_tables).",
    )
    p.add_argument(
        "--similarities",
        type=str,
        required=False,
        default=None,
        help="Path to compatibility cache directory (defaults to cache/<dataset>/{llm}_{embedding_model}/compatibility).",
    )
    p.add_argument(
        "--retrieved-scores",
        type=str,
        default=None,
        help=(
            "Optional path to per-query retrieved-table scores JSON (list[list[{table, similarity_score}]]). "
            "Defaults to results/<dataset>/{llm}_{embedding_model}/results_dense_retriever/<dataset>_k_<top_k>_with_scores.json "
            "when omitted. If omitted and --data contains dict entries, those are used as scores automatically."
        ),
    )

    p.add_argument(
        "--dataset",
        dest="dataset",
        choices=[dt.value for dt in DatabaseType],
        default=DatabaseType.BIRD.value,
        help="Dataset used for cache/results path resolution.",
    )
    # Backward-compatible alias.
    p.add_argument(
        "--database-type",
        dest="dataset",
        choices=[dt.value for dt in DatabaseType],
        default=DatabaseType.BIRD.value,
        help="(deprecated) Same as --dataset.",
    )
    p.add_argument("--llm-model", type=str, default="openai:gpt-4o-mini", help="LLM id for LangChain init_chat_model.")
    p.add_argument(
        "--embedding-model",
        type=str,
        default="fireworks:WhereIsAI/UAE-Large-V1",
        help="Embedding model tag used for cache/results run folder naming.",
    )
    p.add_argument("--temperature", type=float, default=0.0, help="LLM temperature.")
    p.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Only used to derive default --data path when --data is omitted (default: 10).",
    )

    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Similarity threshold used by the original logic for augmentation decisions.",
    )
    p.add_argument(
        "--augment-with-compatible-tables",
        action="store_true",
        default=False,
        help="Apply augmentation after selection.",
    )
    p.add_argument("--no-cache", action="store_true", default=False, help="Disable reading/writing selection cache.")

    p.add_argument("--start", type=int, default=0, help="Start index in the dataset (default: 0).")
    p.add_argument("--limit", type=int, default=None, help="Optional limit on number of queries to process.")
    p.add_argument(
        "--partition",
        type=int,
        default=None,
        help=(
            "Optional partition index (0-based) for contiguous partitioning. "
            "Use together with --num-partitions."
        ),
    )
    p.add_argument(
        "--num-partitions",
        type=int,
        default=None,
        help="Optional number of contiguous partitions. Use together with --partition.",
    )
    p.add_argument(
        "--merge-partitions-only",
        action="store_true",
        default=False,
        help=(
            "If set, skip table selection and only merge partition output files "
            "(*_i_of_N.json) into the base output path. Requires --num-partitions."
        ),
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help=(
            "Randomly sample this many queries from the dataset before processing. "
            "Sampling is reproducible via --random-seed."
        ),
    )
    p.add_argument("--random-seed", type=int, default=42, help="Random seed used for --num-samples.")
    p.add_argument(
        "--out-details",
        type=str,
        default=None,
        help="Optional path to write a verbose per-query output (debug/details).",
    )
    p.add_argument(
        "--write-details",
        action="store_true",
        default=True,
        help="If set, write details to the default results folder even when --out-details is not provided.",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path (defaults to results/<dataset>/{llm}_{embedding_model}/results_table_selection/selected_tables.json).",
    )
    p.add_argument("--indent", type=int, default=2, help="JSON indentation level.")
    p.add_argument(
        "--compat-use-csv-fastpath",
        action="store_true",
        default=True,
        help=(
            "If set, online compatibility calculation on cache miss uses CSV-first mode "
            "(then SQL fallback), matching compatibility_calculator's --use-csv-fastpath behavior."
        ),
    )
    p.add_argument(
        "--compat-max-cached-table-profiles",
        type=int,
        default=463,
        help="Max in-memory cached CSV table profiles for online compatibility calculation.",
    )
    return p.parse_args(argv)


def _partition_bounds(total: int, *, partition: int, num_partitions: int) -> Tuple[int, int]:
    """
    Compute contiguous [start, end) bounds for `partition` out of `num_partitions`.

    Uses floor boundaries:
      start = floor(partition * total / num_partitions)
      end   = floor((partition + 1) * total / num_partitions)
    """
    if num_partitions <= 0:
        raise ValueError(f"num_partitions must be > 0 (got {num_partitions})")
    if partition < 0 or partition >= num_partitions:
        raise ValueError(f"partition must be in [0, {num_partitions - 1}] (got {partition})")
    if total <= 0:
        return (0, 0)
    start = (partition * total) // num_partitions
    end = ((partition + 1) * total) // num_partitions
    return int(start), int(end)


def _append_partition_suffix(path: Path, *, partition: int, num_partitions: int) -> Path:
    """
    Append `_partition_of_num_partitions` to the filename (before extension).

    Example:
      selected_tables.json -> selected_tables_0_of_2.json
    """
    suffix = f"_{int(partition)}_of_{int(num_partitions)}"
    if path.suffix:
        return path.with_name(f"{path.stem}{suffix}{path.suffix}")
    return path.with_name(f"{path.name}{suffix}")


def _merge_partition_json_lists(base_path: Path, *, num_partitions: int) -> Tuple[List[Any], int]:
    """
    Merge list-valued partition JSON files into a single list.

    Reads files named like:
      <base_stem>_<partition>_of_<num_partitions><suffix>
    """
    merged: List[Any] = []
    files_found = 0
    for part in range(int(num_partitions)):
        part_path = _append_partition_suffix(base_path, partition=part, num_partitions=num_partitions)
        if not part_path.exists():
            print(f"⚠️  Missing partition file, skipping: {part_path}")
            continue
        try:
            payload = _read_json(part_path)
            if isinstance(payload, list):
                merged.extend(payload)
                files_found += 1
            else:
                print(f"⚠️  Partition file is not a list, skipping: {part_path}")
        except Exception as e:
            print(f"⚠️  Failed to read partition file '{part_path}': {e}")
    return merged, files_found


def _normalize_table_id(table_id: Any) -> str:
    s = str(table_id or "").strip()
    if "#sep#" in s:
        dbp, tp = s.split("#sep#", 1)
        return f"{dbp.strip().lower()}#sep#{tp.strip().lower()}"
    return s.lower()


def _default_queries_path(database_type: DatabaseType) -> Path:
    return (PROJECT_ROOT / "data" / database_type.value / "dev.json").resolve()


def _load_queries(path: Path) -> List[Dict[str, Any]]:
    try:
        payload = _read_json(path)
        return payload if isinstance(payload, list) else []
    except Exception as e:
        print(f"❌ Failed to read queries file '{path}': {e}")
        return []


def _load_retrieved_tables(path: Path) -> List[List[str]]:
    try:
        payload = _read_json(path)
        if not isinstance(payload, list):
            return []
        out: List[List[str]] = []
        for entry in payload:
            if isinstance(entry, list):
                # Accept either list[str] or list[{"table": "...", ...}] (dense retriever with scores).
                normed: List[str] = []
                for item in entry:
                    if isinstance(item, str):
                        normed.append(_normalize_table_id(item))
                    elif isinstance(item, dict) and "table" in item:
                        normed.append(_normalize_table_id(item.get("table")))
                out.append([t for t in normed if t])
            else:
                out.append([])
        return out
    except Exception as e:
        print(f"❌ Failed to read data file '{path}': {e}")
        return []


def _load_retrieved_scores(path: Optional[Path]) -> Optional[List[List[Dict[str, Any]]]]:
    if path is None:
        return None
    try:
        payload = _read_json(path)
        if not isinstance(payload, list):
            return None
        out: List[List[Dict[str, Any]]] = []
        for entry in payload:
            if isinstance(entry, list):
                out.append([x for x in entry if isinstance(x, dict)])
            else:
                out.append([])
        return out
    except Exception as e:
        print(f"⚠️  Failed to read retrieved-scores file '{path}': {e}")
        return None


def _maybe_scores_from_data_file(path: Path) -> Optional[List[List[Dict[str, Any]]]]:
    """If `--data` contains dict entries, reuse them as scores payload."""
    try:
        payload = _read_json(path)
        if not isinstance(payload, list):
            return None
        out: List[List[Dict[str, Any]]] = []
        saw_any = False
        for entry in payload:
            if isinstance(entry, list):
                dicts = [x for x in entry if isinstance(x, dict)]
                if dicts:
                    saw_any = True
                out.append(dicts)
            else:
                out.append([])
        return out if saw_any else None
    except Exception:
        return None


def _table_id_from_enriched(enriched: Dict[str, Any]) -> str:
    db_id = str(enriched.get("db_id", "")).strip().lower()
    table_name = str(enriched.get("table_name", "")).strip().lower()
    return f"{db_id}#sep#{table_name}" if db_id and table_name else table_name


def _parse_table_identifier(table_id: str) -> Tuple[str, str]:
    normalized = _normalize_table_id(table_id)
    if "#sep#" in normalized:
        db_id, table_name = normalized.split("#sep#", 1)
        return db_id.strip().lower(), table_name.strip().lower()
    return "", normalized.strip().lower()


def _load_enriched_table_from_cache(enriched_dir: Path, table_id: str) -> Dict[str, Any]:
    """
    Load one enriched table entry from cache.

    Cache file format example:
      { "metadata": { "original_table": ..., "db_id": ..., "table_name": ..., ... }, ... }
    """
    normalized = _normalize_table_id(table_id)
    path = (enriched_dir / f"{normalized}.json").resolve()
    if not path.exists():
        # Fallback: try original string as-is (some callers may not normalize)
        alt = (enriched_dir / f"{str(table_id).strip()}.json").resolve()
        path = alt if alt.exists() else path
    try:
        raw = _read_json(path)
        if isinstance(raw, dict) and isinstance(raw.get("metadata"), dict):
            meta = dict(raw["metadata"])
        elif isinstance(raw, dict):
            meta = dict(raw)
        else:
            meta = {}
    except Exception:
        meta = {}

    if not meta:
        # Minimal placeholder to keep indices stable
        dbp = ""
        tp = normalized
        if "#sep#" in normalized:
            dbp, tp = normalized.split("#sep#", 1)
        meta = {
            "db_id": dbp,
            "table_name": tp,
            "original_table": f"Table name: {tp}\nExample table content:\n| (missing cached enrichment) |",
            "purpose": None,
            "summary": None,
            "qa_pairs": [],
        }

    # Ensure identity fields exist and are normalized
    meta["db_id"] = str(meta.get("db_id") or "").strip().lower()
    meta["table_name"] = str(meta.get("table_name") or "").strip().lower()
    if "original_table" not in meta:
        meta["original_table"] = f"Table name: {meta['table_name']}\nExample table content:\n| |"
    return meta


def _load_compatibility_details(
    compat_dir: Path,
    table_a: str,
    table_b: str,
    *,
    config: Optional[Configuration] = None,
    compat_calculator: Optional[Any] = None,
    compat_storage: Optional[Any] = None,
) -> Dict[str, Any]:
    a = _normalize_table_id(table_a)
    b = _normalize_table_id(table_b)
    cand1 = (compat_dir / f"{a}-{b}.json").resolve()
    cand2 = (compat_dir / f"{b}-{a}.json").resolve()
    path = cand1 if cand1.exists() else (cand2 if cand2.exists() else None)
    if path is not None:
        try:
            raw = _read_json(path)
            return raw if isinstance(raw, dict) else {}
        except Exception:
            return {}

    if compat_calculator is None or config is None:
        return {
            "overall_compatibility": 0.0,
            "best_join_columns": ["", ""],
            "join_confidence": 0.0,
        }

    db_a, table_name_a = _parse_table_identifier(a)
    db_b, table_name_b = _parse_table_identifier(b)
    if not db_a or not db_b or not table_name_a or not table_name_b:
        return {
            "overall_compatibility": 0.0,
            "best_join_columns": ["", ""],
            "join_confidence": 0.0,
        }

    try:
        _ = compat_calculator.calculate_table_compatibility(
            {"db_id": db_a, "table_name": table_name_a},
            {"db_id": db_b, "table_name": table_name_b},
        )
        if compat_storage is not None:
            cached_payload = compat_storage.get_cached_compatibility_score(a, b)
            if isinstance(cached_payload, dict) and cached_payload:
                return cached_payload
        # Fallback: read from file if storage lookup is unavailable.
        path_after = cand1 if cand1.exists() else (cand2 if cand2.exists() else None)
        if path_after is not None:
            try:
                raw = _read_json(path_after)
                if isinstance(raw, dict):
                    return raw
            except Exception:
                pass
        return {
            "overall_compatibility": 0.0,
            "best_join_columns": ["", ""],
            "join_confidence": 0.0,
        }
    except Exception as e:
        print(f"⚠️  Failed online compatibility calculation for {a} <-> {b}: {e}")
        return {
            "overall_compatibility": 0.0,
            "best_join_columns": ["", ""],
            "join_confidence": 0.0,
        }


def _build_similarities_for_tables(
    compat_dir: Path,
    table_ids: List[str],
    *,
    config: Optional[Configuration] = None,
    compat_calculator: Optional[Any] = None,
    compat_storage: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    sims: List[Dict[str, Any]] = []
    for i in range(len(table_ids)):
        for j in range(i + 1, len(table_ids)):
            details = _load_compatibility_details(
                compat_dir,
                table_ids[i],
                table_ids[j],
                config=config,
                compat_calculator=compat_calculator,
                compat_storage=compat_storage,
            )
            try:
                sim_val = float(details.get("overall_compatibility", 0.0))
            except Exception:
                sim_val = 0.0
            sims.append(
                {
                    "table1_index": i,
                    "table2_index": j,
                    "similarity": sim_val,
                    "compatibility_details": details,
                }
            )
    return sims


def _make_config(args: argparse.Namespace) -> Configuration:
    return Configuration(
        database_type=DatabaseType(args.dataset),
        llm_model=str(args.llm_model),
        embedding_model=str(getattr(args, "embedding_model", "fireworks:WhereIsAI/UAE-Large-V1")),
        temperature=float(args.temperature),
    )


def main(args: argparse.Namespace) -> int:
    config = _make_config(args)

    # Default dense-retriever output paths derived from dataset + model tags.
    run_dir = get_results_run_dir(
        dataset=str(args.dataset),
        step_dirname="results_dense_retriever",
        llm_model=str(args.llm_model),
        embedding_model=str(args.embedding_model),
        project_root=PROJECT_ROOT,
    )

    if args.data:
        data_path = Path(args.data).expanduser().resolve()
    else:
        data_path = (run_dir / f"{str(args.dataset)}_k_{int(args.top_k)}.json").resolve()

    # Resolve output paths early so merge-only mode can run without selection.
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        out_dir = get_results_run_dir(
            dataset=str(config.database_type.value),
            step_dirname="results_table_selection",
            llm_model=str(config.llm_model),
            embedding_model=str(getattr(config, "embedding_model", "unknown")),
            project_root=PROJECT_ROOT,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = (out_dir / "selected_tables.json").resolve()

    out_details_path = (
        Path(args.out_details).expanduser().resolve()
        if args.out_details
        else out_path.with_name("selected_tables_details.json")
    )

    if bool(getattr(args, "merge_partitions_only", False)):
        if getattr(args, "num_partitions", None) is None:
            print("❌ --merge-partitions-only requires --num-partitions.")
            return 1
        if int(args.num_partitions) <= 0:
            print("❌ --num-partitions must be > 0 for merge mode.")
            return 1

        merged_selected, files_found = _merge_partition_json_lists(out_path, num_partitions=int(args.num_partitions))
        if files_found == 0:
            print("❌ No partition result files found to merge.")
            return 1
        _write_json(out_path, merged_selected, indent=int(args.indent))
        print(
            f"Merged {files_found}/{int(args.num_partitions)} partition files "
            f"into selected tables: {out_path} (rows={len(merged_selected)})"
        )

        if args.out_details or bool(args.write_details):
            merged_details, details_files_found = _merge_partition_json_lists(
                out_details_path, num_partitions=int(args.num_partitions)
            )
            if details_files_found > 0:
                _write_json(out_details_path, merged_details, indent=int(args.indent))
                print(
                    f"Merged {details_files_found}/{int(args.num_partitions)} partition files "
                    f"into details: {out_details_path} (rows={len(merged_details)})"
                )
            else:
                print("ℹ️  No details partition files found to merge.")
        return 0

    # If --retrieved-scores is omitted, point it to the dense-retriever scores file.
    if args.retrieved_scores:
        retrieved_scores_path: Optional[Path] = Path(args.retrieved_scores).expanduser().resolve()
    else:
        retrieved_scores_path = (run_dir / f"{str(args.dataset)}_k_{int(args.top_k)}_with_scores.json").resolve()
    enriched_dir = (
        Path(args.enriched_tables).expanduser().resolve()
        if args.enriched_tables
        else (Path(config.get_database_cache_dir("metadata")) / "enriched_tables").resolve()
    )
    compat_dir = (
        Path(args.similarities).expanduser().resolve()
        if args.similarities
        else Path(config.get_database_cache_dir("compatibility")).expanduser().resolve()
    )
    compat_dir.mkdir(parents=True, exist_ok=True)

    database_type = DatabaseType(args.dataset)
    queries_path = Path(args.queries_file).expanduser().resolve() if args.queries_file else _default_queries_path(database_type)

    queries = _load_queries(queries_path)
    retrieved_tables_per_query = _load_retrieved_tables(data_path)
    scores_payload = None
    if retrieved_scores_path is not None:
        if retrieved_scores_path.exists():
            scores_payload = _load_retrieved_scores(retrieved_scores_path)
        else:
            # Avoid a noisy warning when the default scores file doesn't exist yet.
            if not args.retrieved_scores:
                print(f"ℹ️  No retrieved-scores file found at default path: {retrieved_scores_path}")
            else:
                print(f"⚠️  Retrieved-scores file not found: {retrieved_scores_path}")
    if scores_payload is None:
        scores_payload = _maybe_scores_from_data_file(data_path)

    if not retrieved_tables_per_query:
        print("❌ No retrieved tables found in --data.")
        return 1
    if not queries:
        print(f"❌ No queries found in queries file: {queries_path}")
        return 1

    n = min(len(queries), len(retrieved_tables_per_query))
    # Choose which indices to process
    all_indices = list(range(n))
    selected_indices = all_indices
    if args.num_samples is not None:
        k = int(args.num_samples)
        if k <= 0:
            print("Nothing to do: --num-samples must be > 0")
            return 0
        if k < n:
            import random

            random.seed(int(args.random_seed))
            selected_indices = sorted(random.sample(range(n), k))
        else:
            selected_indices = all_indices

    # Partitioning (optional): split `selected_indices` into contiguous partitions, then apply --start/--limit within that partition.
    base_positions = list(range(len(selected_indices)))
    part_start, part_end = 0, len(base_positions)
    if getattr(args, "partition", None) is not None or getattr(args, "num_partitions", None) is not None:
        if getattr(args, "partition", None) is None or getattr(args, "num_partitions", None) is None:
            print("❌ Partitioning requires both --partition and --num-partitions.")
            return 1
        try:
            part_start, part_end = _partition_bounds(
                len(base_positions),
                partition=int(args.partition),
                num_partitions=int(args.num_partitions),
            )
        except Exception as e:
            print(f"❌ Invalid partition configuration: {e}")
            return 1

    partition_positions = base_positions[part_start:part_end]

    start = max(0, int(args.start or 0))
    end_local = len(partition_positions) if args.limit is None else min(len(partition_positions), start + int(args.limit))
    if start >= end_local:
        print(
            f"Nothing to do: start={start} end={end_local} "
            f"(partition_size={len(partition_positions)} selected={len(selected_indices)})"
        )
        return 0

    # Positions into `selected_indices` we will actually process.
    positions = partition_positions[start:end_local]

    selected_tables_out: List[List[str]] = []
    details_out: List[Dict[str, Any]] = []

    print(f"Total available queries: {n}")
    if args.num_samples is not None:
        print(f"Random sampling enabled: num_samples={int(args.num_samples)} seed={int(args.random_seed)} -> selected={len(selected_indices)}")
    if getattr(args, "partition", None) is not None and getattr(args, "num_partitions", None) is not None:
        # part_end is exclusive
        print(
            f"Partitioning enabled: partition={int(args.partition)}/{int(args.num_partitions)} "
            f"-> global positions {part_start}..{max(part_end - 1, part_start - 1)} (count={len(partition_positions)})"
        )
    print(f"Processing partition-local slice {start}..{end_local-1} (count={end_local-start})")

    compat_calculator = None
    compat_storage = None
    try:
        from offline_preprocessing.compatibility_calculator import CompatibilityCalculator
        from utils.storage_manager import UnifiedStorageManager

        compat_cache_root = compat_dir.parent
        storage = UnifiedStorageManager(config=config, cache_dir=str(compat_cache_root))
        compat_storage = storage
        compat_calculator = CompatibilityCalculator(
            config=config,
            storage=storage,
            embedding_model=str(getattr(config, "embedding_model", "openai:text-embedding-3-large")),
            use_csv_fastpath=bool(getattr(args, "compat_use_csv_fastpath", False)),
            max_cached_table_profiles=max(1, int(getattr(args, "compat_max_cached_table_profiles", 8))),
        )
    except Exception as e:
        print(f"⚠️  Could not initialize online compatibility calculator: {e}")
        compat_calculator = None

    shared_model: Optional[Any] = None

    for pos in positions:
        idx = selected_indices[pos]
        q = queries[idx] if 0 <= idx < len(queries) else {}
        query_text = str(q.get("question", "") or "")
        db_id = str(q.get("db_id", "") or "").strip().lower()
        retrieved_tables = retrieved_tables_per_query[idx] if idx < len(retrieved_tables_per_query) else []
        retrieved_tables = [_normalize_table_id(t) for t in (retrieved_tables or [])]

        # Build scores dict aligned to retrieved tables
        retrieved_scores: Dict[str, float] = {}
        if scores_payload is not None and 0 <= idx < len(scores_payload):
            for entry in scores_payload[idx] or []:
                try:
                    t_full = _normalize_table_id(entry.get("table", ""))
                    score_val = float(entry.get("similarity_score", 0.0))
                    retrieved_scores[t_full] = score_val
                except Exception:
                    continue

        # Load enriched table objects from cache (one file per table)
        enriched_tables: List[Dict[str, Any]] = []
        for t in retrieved_tables:
            meta = _load_enriched_table_from_cache(enriched_dir, t)
            # Ensure db_id aligns with query db_id when missing
            if not meta.get("db_id") and db_id and "#sep#" in t:
                meta["db_id"] = db_id
            enriched_tables.append(meta)

        # Load pairwise compatibilities from cache and convert to similarity list with indices
        similarities = _build_similarities_for_tables(
            compat_dir,
            retrieved_tables,
            config=config,
            compat_calculator=compat_calculator,
            compat_storage=compat_storage,
        )

        clustered_tables, cluster_summary, _, shared_model = validate_and_cluster_tables(
            enriched_tables,
            similarities,
            query_text,
            config=config,
            retrieved_tables_scores=retrieved_scores or None,
            augment_with_compatible_tables=bool(args.augment_with_compatible_tables),
            threshold=float(args.threshold),
            use_cache=not bool(args.no_cache),
            model=shared_model,
        )

        selected_tables: List[str] = []
        if clustered_tables and isinstance(clustered_tables, list) and isinstance(clustered_tables[0], dict):
            idxs = clustered_tables[0].get("table_indices") or []
            if isinstance(idxs, list):
                for ti in idxs:
                    try:
                        ti_int = int(ti)
                    except Exception:
                        continue
                    if 0 <= ti_int < len(enriched_tables):
                        selected_tables.append(_table_id_from_enriched(enriched_tables[ti_int]))

        selected_tables_out.append(selected_tables)

        if args.out_details or bool(args.write_details):
            # Break down final selection into: LLM-only selection and augmentation-only additions.
            llm_selected_tables: List[str] = []
            augmented_tables_only: List[str] = []
            try:
                llm_idxs = cluster_summary.get("llm_selected_table_indices", []) if isinstance(cluster_summary, dict) else []
                aug_only_idxs = cluster_summary.get("augmented_only_table_indices", []) if isinstance(cluster_summary, dict) else []

                if isinstance(llm_idxs, list) and llm_idxs:
                    for ti in llm_idxs:
                        try:
                            ti_int = int(ti)
                        except Exception:
                            continue
                        if 0 <= ti_int < len(enriched_tables):
                            llm_selected_tables.append(_table_id_from_enriched(enriched_tables[ti_int]))

                if isinstance(aug_only_idxs, list) and aug_only_idxs:
                    for ti in aug_only_idxs:
                        try:
                            ti_int = int(ti)
                        except Exception:
                            continue
                        if 0 <= ti_int < len(enriched_tables):
                            augmented_tables_only.append(_table_id_from_enriched(enriched_tables[ti_int]))

                # If we couldn't infer pre-augmentation selection, fall back to final.
                if not llm_selected_tables:
                    llm_selected_tables = list(selected_tables)

                # Ensure augmented_only is consistent with the two lists.
                llm_set = set(llm_selected_tables)
                augmented_tables_only = [t for t in augmented_tables_only if t and t not in llm_set]
            except Exception:
                llm_selected_tables = list(selected_tables)
                augmented_tables_only = []

            details_out.append(
                {
                    "question_id": idx,
                    "sample_position": pos,
                    "db_id": db_id,
                    "query": query_text,
                    "retrieved_tables": retrieved_tables,
                    "llm_selected_tables": llm_selected_tables,
                    "augmented_tables_only": augmented_tables_only,
                    "selected_tables": selected_tables,
                    "cluster_summary": cluster_summary,
                }
            )

        print(f"  - processed question_id={idx} (selected={len(selected_tables)})")

    # If partitioning is enabled, suffix outputs with `{partition}_of_{num_partitions}`.
    if getattr(args, "partition", None) is not None and getattr(args, "num_partitions", None) is not None:
        out_path = _append_partition_suffix(
            out_path,
            partition=int(args.partition),
            num_partitions=int(args.num_partitions),
        )
    _write_json(out_path, selected_tables_out, indent=int(args.indent))
    print(f"Wrote selected tables: {out_path}")

    if args.out_details or bool(args.write_details):
        out_details_path = (
            Path(args.out_details).expanduser().resolve()
            if args.out_details
            else out_path.with_name("selected_tables_details.json")
        )
        if getattr(args, "partition", None) is not None and getattr(args, "num_partitions", None) is not None:
            out_details_path = _append_partition_suffix(
                out_details_path,
                partition=int(args.partition),
                num_partitions=int(args.num_partitions),
            )
        _write_json(out_details_path, details_out, indent=int(args.indent))
        print(f"Wrote details: {out_details_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(_parse_args()))

