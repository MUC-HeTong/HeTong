"""Current-KB, source-only, direction-aware exact terminology retrieval."""

from __future__ import annotations

import unicodedata
from collections import defaultdict
from copy import deepcopy
from typing import Any, Iterable


SCHEMA_VERSION = "current_kb_exact_retrieval_v1"
INDEX_SCHEMA_VERSION = "current_kb_exact_index_v1"
RETRIEVER_INPUT_FIELDS = frozenset({"record_id", "direction", "source"})
SUPPORTED_DIRECTIONS = frozenset({"bo2zh", "zh2bo"})


def normalized_length(value: str) -> int:
    return sum(not unicodedata.category(char).startswith(("P", "Z")) for char in value)


def to_retriever_input(row: dict[str, Any]) -> dict[str, str]:
    return {field: row[field] for field in ("record_id", "direction", "source")}


def require_retriever_input(record: dict[str, Any]) -> None:
    if set(record) != RETRIEVER_INPUT_FIELDS:
        raise ValueError("retriever input must contain exactly record_id, direction, and source")
    if record["direction"] not in SUPPORTED_DIRECTIONS:
        raise ValueError(f"unsupported direction: {record['direction']}")
    if not isinstance(record["record_id"], (str, int)) or not isinstance(record["source"], str):
        raise ValueError("record_id must be string/int and source must be string")


def _entry_payload(entry: dict[str, Any], target_term: str, match_types: set[str]) -> dict[str, Any]:
    return {
        "term_id": entry["term_id"],
        "target_term": target_term,
        "bo_canonical": entry["bo_canonical"],
        "zh_term": entry["zh_term"],
        "bo_surface_forms": sorted(set(entry.get("bo_surface_forms", [])) | {entry["bo_canonical"]}),
        "conflict_status": entry["conflict_status"],
        "needs_review": bool(entry["needs_review"]),
        "source_count": int(entry.get("source_count", 0)),
        "source_pair_count": int(entry.get("source_pair_count", 0)),
        "source_files": sorted(entry.get("source_files", [])),
        "match_types": sorted(match_types),
    }


def build_index(
    entries: Iterable[dict[str, Any]],
    aliases: Iterable[dict[str, Any]] | None = None,
    include_observed_surfaces: bool = True,
) -> dict[str, Any]:
    """Build JSON-serializable surface indexes using only current KB records."""
    by_id = {entry["term_id"]: deepcopy(entry) for entry in entries}
    bo_surfaces: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for entry in by_id.values():
        canonical = entry["bo_canonical"]
        bo_surfaces[canonical][entry["term_id"]].add("canonical")
        if include_observed_surfaces:
            for surface in entry.get("bo_surface_forms", []):
                bo_surfaces[surface][entry["term_id"]].add("canonical" if surface == canonical else "alias")
    if aliases is not None and include_observed_surfaces:
        for alias in aliases:
            term_id, surface = alias["term_id"], alias["bo_surface_form"]
            if term_id not in by_id:
                raise ValueError(f"alias references unknown term_id: {term_id}")
            bo_surfaces[surface][term_id].add("canonical" if surface == by_id[term_id]["bo_canonical"] else "alias")

    def materialize(mapping: dict[str, dict[str, set[str]]], direction: str) -> dict[str, list[dict[str, Any]]]:
        output: dict[str, list[dict[str, Any]]] = {}
        for surface, item_map in mapping.items():
            output[surface] = sorted(
                [_entry_payload(by_id[term_id], by_id[term_id]["zh_term"] if direction == "bo2zh" else by_id[term_id]["bo_canonical"], kinds)
                 for term_id, kinds in item_map.items()],
                key=lambda item: item["term_id"],
            )
        return dict(sorted(output.items()))

    zh_surfaces: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for entry in by_id.values():
        zh_surfaces[entry["zh_term"]][entry["term_id"]].add("chinese_exact")
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "bo2zh": materialize(bo_surfaces, "bo2zh"),
        "zh2bo": materialize(zh_surfaces, "zh2bo"),
    }


def _surface_decision(surface: str, start: int, source: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    target_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        target_groups[candidate["target_term"]].append(candidate)
    targets = sorted(target_groups)
    has_bidirectional = any(item["conflict_status"] == "BIDIRECTIONAL_CONFLICT" for item in candidates)
    if has_bidirectional:
        status = "BIDIRECTIONAL_CONFLICT_BLOCKED"
    elif len(targets) != 1:
        status = "AMBIGUOUS_MATCH"
    else:
        status = "HIT"
    selected = sorted(target_groups[targets[0]], key=lambda item: item["term_id"])[0] if len(targets) == 1 else None
    selected_term_id = selected["term_id"] if selected else ""
    kinds = sorted({kind for item in candidates for kind in item["match_types"]})
    match_type = "canonical" if "canonical" in kinds else "alias" if "alias" in kinds else "Chinese exact"
    return {
        "retrieval_status": status,
        "matched_source_surface": surface,
        "matched_span": {"start": start, "end": start + len(surface)},
        "match_type": match_type,
        "match_length": normalized_length(surface),
        "raw_match_length": len(surface),
        "source_coverage_ratio": normalized_length(surface) / max(1, normalized_length(source)),
        "candidate_entry_count": len(candidates),
        "distinct_target_count": len(targets),
        "candidate_targets": targets,
        "source_entry_ids": [item["term_id"] for item in candidates],
        "conflict_statuses": sorted({item["conflict_status"] for item in candidates}),
        "would_be_unique_without_bidirectional_block": has_bidirectional and len(targets) == 1,
        "selected": selected,
        "rank_key": (
            -normalized_length(surface),
            -len(surface),
            -normalized_length(surface) / max(1, normalized_length(source)),
            -max(item["source_count"] for item in candidates),
            -max(item["source_pair_count"] for item in candidates),
            start,
            surface,
            selected_term_id,
        ),
    }


def _result(record: dict[str, Any], decision: dict[str, Any] | None, matched_surface_count: int) -> dict[str, Any]:
    base = {
        "schema_version": SCHEMA_VERSION,
        "record_id": str(record["record_id"]),
        "direction": record["direction"],
        "source": record["source"],
        "strategy": "single",
        "matched_surface_count": matched_surface_count,
    }
    if decision is None:
        return base | {
            "retrieval_status": "NO_HIT", "matched_source_surface": None, "matched_span": None,
            "match_type": None, "match_length": 0, "raw_match_length": 0, "candidate_entry_count": 0, "distinct_target_count": 0,
            "candidate_targets": [], "source_entry_ids": [], "conflict_statuses": [],
            "would_be_unique_without_bidirectional_block": False, "target_term": None, "terminology_id": None,
            "canonical_target_term": None, "surface_target_terms": [],
        }
    selected = decision["selected"] if decision["retrieval_status"] == "HIT" else None
    return base | {
        key: decision[key] for key in (
            "retrieval_status", "matched_source_surface", "matched_span", "match_type", "match_length", "raw_match_length",
            "candidate_entry_count", "distinct_target_count", "candidate_targets", "source_entry_ids",
            "conflict_statuses", "would_be_unique_without_bidirectional_block",
        )
    } | {
        "target_term": selected["target_term"] if selected else None,
        "terminology_id": selected["term_id"] if selected else None,
        "canonical_target_term": selected["zh_term"] if selected and record["direction"] == "bo2zh" else selected["bo_canonical"] if selected else None,
        "surface_target_terms": [selected["zh_term"]] if selected and record["direction"] == "bo2zh" else selected["bo_surface_forms"] if selected else [],
    }


def retrieve(record: dict[str, Any], index: dict[str, Any]) -> dict[str, Any]:
    require_retriever_input(record)
    source_map = index[record["direction"]]
    decisions: list[dict[str, Any]] = []
    for surface, candidates in source_map.items():
        start = record["source"].find(surface)
        while start >= 0:
            decisions.append(_surface_decision(surface, start, record["source"], candidates))
            start = record["source"].find(surface, start + len(surface))
    # A longer or otherwise higher-ranked overlapping surface owns the span.
    # This prevents a canonical prefix from bypassing ambiguity on a matched alias.
    retained: list[dict[str, Any]] = []
    for decision in sorted(decisions, key=lambda item: item["rank_key"]):
        start = decision["matched_span"]["start"]
        end = decision["matched_span"]["end"]
        if any(start < kept["matched_span"]["end"] and kept["matched_span"]["start"] < end for kept in retained):
            continue
        retained.append(decision)
    hits = [item for item in retained if item["retrieval_status"] == "HIT"]
    if hits:
        return _result(record, min(hits, key=lambda item: item["rank_key"]), len(decisions))
    non_hits = [item for item in retained if item["retrieval_status"] != "HIT"]
    return _result(record, min(non_hits, key=lambda item: item["rank_key"]) if non_hits else None, len(decisions))


def audit_index(index: dict[str, Any]) -> dict[str, dict[str, int]]:
    audit: dict[str, dict[str, int]] = {}
    for direction in ("bo2zh", "zh2bo"):
        surfaces = index[direction]
        multi = []
        bidirectional = []
        canonical_ambiguities = []
        alias_ambiguities = []
        unique = 0
        for surface, candidates in surfaces.items():
            target_count = len({candidate["target_term"] for candidate in candidates})
            if target_count == 1:
                unique += 1
            else:
                multi.append(surface)
                kinds = {kind for candidate in candidates for kind in candidate["match_types"]}
                if direction == "bo2zh" and "alias" in kinds:
                    alias_ambiguities.append(surface)
                if direction == "zh2bo" or "canonical" in kinds:
                    canonical_ambiguities.append(surface)
            if any(candidate["conflict_status"] == "BIDIRECTIONAL_CONFLICT" for candidate in candidates):
                bidirectional.append(surface)
        audit[direction] = {
            "unique_source_surfaces": len(surfaces),
            "source_surfaces_with_exactly_one_target": unique,
            "source_surfaces_with_multiple_targets": len(multi),
            "canonical_source_ambiguities": len(canonical_ambiguities),
            "alias_source_ambiguities": len(alias_ambiguities),
            "source_surfaces_with_bidirectional_conflict": len(bidirectional),
            "bidirectional_would_be_unique_by_surface": sum(
                len({candidate["target_term"] for candidate in surfaces[surface]}) == 1 for surface in bidirectional
            ),
        }
    return audit
