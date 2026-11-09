"""Source-only exact-span candidates and deterministic ranking policies."""

from __future__ import annotations

import unicodedata
from typing import Any

from current_kb_exact_retriever import _surface_decision, normalized_length, require_retriever_input


POLICY_FEATURES = {
    "R1": ("normalized_length_desc", "raw_length_desc", "coverage_desc", "source_count_desc", "source_pair_count_desc", "start_asc", "surface_asc", "term_id_asc"),
    "R2": ("normalized_length_asc", "raw_length_asc", "coverage_desc", "source_count_desc", "source_pair_count_desc", "start_asc", "surface_asc", "term_id_asc"),
    "R3": ("canonical_first", "normalized_length_desc", "raw_length_desc", "coverage_desc", "source_count_desc", "source_pair_count_desc", "start_asc", "surface_asc", "term_id_asc"),
    "R4": ("source_count_desc", "source_pair_count_desc", "normalized_length_desc", "raw_length_desc", "coverage_desc", "start_asc", "surface_asc", "term_id_asc"),
    "R5": ("boundary_complete_desc", "normalized_length_desc", "raw_length_desc", "coverage_desc", "source_count_desc", "source_pair_count_desc", "start_asc", "surface_asc", "term_id_asc"),
    "R6": ("canonical_first", "boundary_complete_desc", "source_count_desc", "source_pair_count_desc", "normalized_length_asc", "raw_length_asc", "coverage_desc", "start_asc", "surface_asc", "term_id_asc"),
}


def _is_boundary(char: str, direction: str) -> bool:
    if char.isspace() or unicodedata.category(char).startswith("P"):
        return True
    if direction == "bo2zh":
        return char in "་།༔"
    return not ("\u4e00" <= char <= "\u9fff")


def boundary_complete(source: str, start: int, end: int, direction: str) -> bool:
    return (start == 0 or _is_boundary(source[start - 1], direction)) and (end == len(source) or _is_boundary(source[end], direction))


def enumerate_candidates(record: dict[str, Any], index: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every raw exact occurrence; no policy or gold fields participate."""
    require_retriever_input(record)
    output = []
    for surface, candidates in index[record["direction"]].items():
        start = record["source"].find(surface)
        while start >= 0:
            decision = _surface_decision(surface, start, record["source"], candidates)
            selected = decision["selected"]
            end = start + len(surface)
            output.append({
                "surface": surface, "start": start, "end": end,
                "retrieval_status": decision["retrieval_status"],
                "match_type": decision["match_type"], "normalized_length": normalized_length(surface),
                "raw_length": len(surface), "coverage": decision["source_coverage_ratio"],
                "source_count": max(item["source_count"] for item in candidates),
                "source_pair_count": max(item["source_pair_count"] for item in candidates),
                "term_id": selected["term_id"] if selected else "",
                "target_term": selected["target_term"] if selected else None,
                "canonical_source_term": selected["bo_canonical"] if selected else None,
                "canonical_target_term": selected["zh_term"] if selected and record["direction"] == "bo2zh" else selected["bo_canonical"] if selected else None,
                "surface_target_terms": [selected["zh_term"]] if selected and record["direction"] == "bo2zh" else selected["bo_surface_forms"] if selected else [],
                "provenance": selected["source_files"] if selected else sorted({f for item in candidates for f in item["source_files"]}),
                "conflict_statuses": decision["conflict_statuses"],
                "candidate_targets": decision["candidate_targets"],
                "candidate_entry_count": decision["candidate_entry_count"],
                "distinct_target_count": decision["distinct_target_count"],
                "boundary_complete": boundary_complete(record["source"], start, end, record["direction"]),
                "r1_rank_key": list(decision["rank_key"]),
            })
            start = record["source"].find(surface, start + len(surface))
    return sorted(output, key=lambda row: (row["start"], row["end"], row["surface"]))


def rank_key(span: dict[str, Any], policy: str) -> tuple[Any, ...]:
    if policy not in POLICY_FEATURES:
        raise ValueError(f"unknown ranking policy: {policy}")
    values = {
        "normalized_length_desc": -span["normalized_length"], "normalized_length_asc": span["normalized_length"],
        "raw_length_desc": -span["raw_length"], "raw_length_asc": span["raw_length"],
        "coverage_desc": -span["coverage"], "source_count_desc": -span["source_count"],
        "source_pair_count_desc": -span["source_pair_count"], "canonical_first": 0 if span["match_type"] == "canonical" else 1,
        "boundary_complete_desc": 0 if span["boundary_complete"] else 1,
        "start_asc": span["start"], "surface_asc": span["surface"], "term_id_asc": span["term_id"],
    }
    return tuple(values[name] for name in POLICY_FEATURES[policy])


def select_candidate(spans: list[dict[str, Any]], policy: str) -> dict[str, Any] | None:
    """Apply policy-specific non-overlap ownership, then select an eligible HIT."""
    retained = []
    for span in sorted(spans, key=lambda value: rank_key(value, policy)):
        if any(span["start"] < prior["end"] and prior["start"] < span["end"] for prior in retained):
            continue
        retained.append(span)
    hits = [span for span in retained if span["retrieval_status"] == "HIT"]
    selected = min(hits, key=lambda value: rank_key(value, policy)) if hits else min(retained, key=lambda value: rank_key(value, policy), default=None)
    if selected is None:
        return None
    return selected | {"rank_key": list(rank_key(selected, policy)), "policy": policy,
                       "valid_terminology_retrieval": selected["retrieval_status"] == "HIT"}
