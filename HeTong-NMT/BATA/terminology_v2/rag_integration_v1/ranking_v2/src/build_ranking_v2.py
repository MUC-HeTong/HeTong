#!/usr/bin/env python3
"""Dev-selected, source-only exact-span ranking experiment; no model inference."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
INTEGRATION = HERE.parents[1]
TERMINOLOGY = INTEGRATION.parent
TERMINOLOGY = Path(os.environ.get("BATA_TERMINOLOGY_ROOT", TERMINOLOGY)).expanduser().resolve()
PROJECT = Path(os.environ.get("BATA_PROJECT_ROOT", TERMINOLOGY.parent)).expanduser().resolve()
sys.path.insert(0, str(INTEGRATION / "src"))
from current_kb_exact_retriever import build_index, normalized_length
from ranking_candidates import POLICY_FEATURES, enumerate_candidates, select_candidate

OUT = INTEGRATION / "ranking_v2"
STEP23R = Path(os.environ.get("BATA_STABILITY_DEV_FILE", PROJECT / "data/stability_dev486.jsonl"))
BEHAVIOR_DEV = {
    "bo2zh": TERMINOLOGY / "stage1_behavior_dataset_v5/bo2zh/dev.jsonl",
    "zh2bo": TERMINOLOGY / "stage1_behavior_dataset_v5/zh2bo/dev.jsonl",
}
FINAL = Path(os.environ.get("BATA_FINAL_DATA_DIR", PROJECT / "data/final2000"))
ENTRIES = TERMINOLOGY / "terminology_entries.jsonl"
ALIASES = TERMINOLOGY / "terminology_aliases.jsonl"
POLICIES = tuple(POLICY_FEATURES)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonicalize_bo(value: str) -> str:
    return " ".join(value.strip().rstrip("།").split())


def derive_dev_records(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    bo, zh = [], []
    for row in rows:
        common = {"record_id": str(row["record_id"]), "reference": row["reference"]}
        bo.append(common | {"direction": "bo2zh", "source": row["input"], "source_term": row["bo_term"], "target_term": row["zh_term"], "bo_term": row["bo_term"], "zh_term": row["zh_term"]})
        zh.append(common | {"direction": "zh2bo", "source": row["reference"], "source_term": row["zh_term"], "target_term": row["bo_term"], "bo_term": row["bo_term"], "zh_term": row["zh_term"]})
    return {"bo2zh": bo, "zh2bo": zh}


def derive_final_records(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {direction: [dict(row) for row in values] for direction, values in rows.items()}


def derive_behavior_dev_records(rows: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    """Use retained direction-native Behavior v5 dev rows; no model output fields."""
    return {direction: [
        {key: row[key] for key in ("record_id", "direction", "source", "reference", "source_term", "target_term", "canonical_target_term", "terminology_id")}
        | {"bo_term": row["source_term"] if direction == "bo2zh" else row["target_term"],
           "zh_term": row["target_term"] if direction == "bo2zh" else row["source_term"]}
        for row in direction_rows
    ] for direction, direction_rows in rows.items()}


def _term_id(row: dict[str, Any], entries_by_pair: dict[tuple[str, str], str]) -> str | None:
    return entries_by_pair.get((row["zh_term"], canonicalize_bo(row["bo_term"])))


def _source_key(row: dict[str, Any]) -> str:
    return canonicalize_bo(row["source"]) if row["direction"] == "bo2zh" else "".join(row["source"].split())


def _span_key(row: dict[str, Any]) -> str:
    value = row["source_term"]
    return canonicalize_bo(value) if row["direction"] == "bo2zh" else "".join(value.split())


def assert_dev_disjoint(dev: dict[str, list[dict[str, Any]]], final: dict[str, list[dict[str, Any]]], entries_by_pair: dict[tuple[str, str], str], raise_on_overlap: bool = True) -> dict[str, Any]:
    audit: dict[str, Any] = {"directions": {}, "eligible_for_policy_selection": True}
    for direction in ("bo2zh", "zh2bo"):
        left, right = dev[direction], final[direction]
        fields = {
            "record_id_overlap": {str(row["record_id"]) for row in left} & {str(row["record_id"]) for row in right},
            "normalized_source_overlap": {_source_key(row) for row in left} & {_source_key(row) for row in right},
            "normalized_source_span_overlap": {_span_key(row) for row in left} & {_span_key(row) for row in right},
            "terminology_pair_overlap": {(canonicalize_bo(row["bo_term"]), row["zh_term"]) for row in left} & {(canonicalize_bo(row["bo_term"]), row["zh_term"]) for row in right},
            "kb_entry_overlap": {_term_id(row, entries_by_pair) for row in left} & {_term_id(row, entries_by_pair) for row in right},
        }
        fields["kb_entry_overlap"].discard(None)
        audit["directions"][direction] = {key: len(value) for key, value in fields.items()}
        if any(fields.values()):
            audit["eligible_for_policy_selection"] = False
    if raise_on_overlap and not audit["eligible_for_policy_selection"]:
        raise ValueError("Dev486 overlaps Final2000 under a required leakage key")
    return audit


def remove_final_overlaps(dev: dict[str, list[dict[str, Any]]], final: dict[str, list[dict[str, Any]]], entries_by_pair: dict[tuple[str, str], str]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Data-isolation filter only: remove any record sharing a prescribed Final key."""
    kept, report = {}, {"directions": {}}
    for direction in ("bo2zh", "zh2bo"):
        final_ids = {str(row["record_id"]) for row in final[direction]}
        final_sources = {_source_key(row) for row in final[direction]}
        final_spans = {_span_key(row) for row in final[direction]}
        final_pairs = {(canonicalize_bo(row["bo_term"]), row["zh_term"]) for row in final[direction]}
        final_terms = {_term_id(row, entries_by_pair) for row in final[direction]}
        final_terms.discard(None)
        retained, removed = [], []
        for row in dev[direction]:
            reasons = []
            if str(row["record_id"]) in final_ids: reasons.append("record_id")
            if _source_key(row) in final_sources: reasons.append("normalized_source")
            if _span_key(row) in final_spans: reasons.append("normalized_source_span")
            if (canonicalize_bo(row["bo_term"]), row["zh_term"]) in final_pairs: reasons.append("terminology_pair")
            if _term_id(row, entries_by_pair) is not None and _term_id(row, entries_by_pair) in final_terms: reasons.append("kb_entry")
            if reasons: removed.append({"record_id": row["record_id"], "reasons": reasons})
            else: retained.append(row)
        kept[direction] = retained
        report["directions"][direction] = {"input_count": len(dev[direction]), "removed_count": len(removed), "kept_count": len(retained), "removed": removed}
    return kept, report


def source_only(row: dict[str, Any]) -> dict[str, str]:
    return {key: row[key] for key in ("record_id", "direction", "source")}


def retrieve_row(record: dict[str, str], index: dict[str, Any], policy: str) -> dict[str, Any]:
    spans = enumerate_candidates(record, index)
    selected = select_candidate(spans, policy)
    if selected is None:
        return record | {"policy": policy, "retrieval_status": "NO_HIT", "target_term": None, "canonical_target_term": None, "surface_target_terms": [], "valid_terminology_retrieval": False, "all_matched_spans": spans}
    return record | {key: selected.get(key) for key in ("policy", "retrieval_status", "surface", "start", "end", "match_type", "target_term", "canonical_target_term", "surface_target_terms", "valid_terminology_retrieval", "rank_key", "conflict_statuses")} | {"all_matched_spans": spans}


def classify_result(result: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    canonical_gold = gold.get("canonical_target_term", canonicalize_bo(gold["target_term"]) if gold.get("direction") == "zh2bo" else gold["target_term"])
    exact = result.get("retrieval_status") == "HIT" and result.get("target_term") == gold["target_term"]
    canonical = result.get("retrieval_status") == "HIT" and (result.get("canonical_target_term") == canonical_gold or gold["target_term"] in result.get("surface_target_terms", []))
    valid = bool(result.get("valid_terminology_retrieval"))
    if valid and not canonical:
        taxonomy = "ALTERNATIVE_VALID_TERM"
    elif valid:
        taxonomy = "ANNOTATED_PAIR_ALIGNED" if exact else "CANONICAL_ANNOTATED_PAIR_ALIGNED"
    elif result.get("retrieval_status") == "AMBIGUOUS_MATCH": taxonomy = "AMBIGUOUS"
    elif result.get("retrieval_status") == "BIDIRECTIONAL_CONFLICT_BLOCKED": taxonomy = "BLOCKED"
    elif result.get("retrieval_status") == "NO_HIT": taxonomy = "NO_HIT"
    else: taxonomy = "TRUE_WRONG_PAIR"
    return {"annotation_exact": exact, "annotation_canonical": canonical, "valid_terminology_retrieval": valid, "taxonomy": taxonomy}


def metrics(results: list[dict[str, Any]], gold: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [classify_result(result, row) for result, row in zip(results, gold)]
    count = len(labels) or 1
    counts = Counter(item["taxonomy"] for item in labels)
    return {"sample_count": len(labels), "annotation_exact_count": sum(item["annotation_exact"] for item in labels), "annotation_canonical_count": sum(item["annotation_canonical"] for item in labels), "valid_count": sum(item["valid_terminology_retrieval"] for item in labels), "valid_rate": sum(item["valid_terminology_retrieval"] for item in labels) / count, "annotation_rate": sum(item["annotation_canonical"] for item in labels) / count, "nonhit_count": sum(counts[name] for name in ("AMBIGUOUS", "BLOCKED", "NO_HIT")), "taxonomy": dict(sorted(counts.items()))}


def select_policy(policy_metrics: dict[str, dict[str, Any]]) -> str:
    return min(policy_metrics, key=lambda name: (-policy_metrics[name]["valid_rate"], -policy_metrics[name]["annotation_rate"], policy_metrics[name]["nonhit_count"], name))


def main() -> None:
    entries, aliases = read_jsonl(ENTRIES), read_jsonl(ALIASES)
    index = build_index(entries, aliases)
    by_pair = {(row["zh_term"], canonicalize_bo(row["bo_canonical"])): row["term_id"] for row in entries}
    final = derive_final_records({direction: read_jsonl(FINAL / f"{direction}_final2000.jsonl") for direction in ("bo2zh", "zh2bo")})
    # Preserve the rejected first-priority Dev486 audit as evidence, then use
    # an independently built retained terminology_v2 dev split after isolation.
    dev486 = derive_dev_records(read_jsonl(STEP23R))
    audit486 = assert_dev_disjoint(dev486, final, by_pair, raise_on_overlap=False)
    write_json(OUT / "manifests/dev486_leakage_audit.json", audit486)
    behavior = derive_behavior_dev_records({direction: read_jsonl(path) for direction, path in BEHAVIOR_DEV.items()})
    dev, filter_report = remove_final_overlaps(behavior, final, by_pair)
    audit = assert_dev_disjoint(dev, final, by_pair, raise_on_overlap=False)
    audit |= {"dataset": "BehaviorV5Dev538", "source_assets": {key: str(value) for key, value in BEHAVIOR_DEV.items()}, "filter_report": filter_report}
    write_json(OUT / "manifests/behavior_v5_dev_leakage_audit.json", audit)
    if not audit["eligible_for_policy_selection"]:
        raise ValueError("BehaviorV5 development set overlaps Final2000 after isolation filter")
    # Source-only outputs are serialized before gold labels participate in metrics.
    dev_outputs: dict[str, dict[str, list[dict[str, Any]]]] = {policy: {} for policy in POLICIES}
    dev_metric: dict[str, dict[str, Any]] = {policy: {} for policy in POLICIES}
    for policy in POLICIES:
        for direction in ("bo2zh", "zh2bo"):
            rows = [retrieve_row(source_only(row), index, policy) for row in dev[direction]]
            path = OUT / "dev" / f"{direction}_{policy}_source_only.jsonl"; write_jsonl(path, rows)
            dev_outputs[policy][direction] = rows
            dev_metric[policy][direction] = metrics(rows, dev[direction])
        combined = {key: sum(dev_metric[policy][direction][key] for direction in ("bo2zh", "zh2bo")) for key in ("sample_count", "annotation_exact_count", "annotation_canonical_count", "valid_count", "nonhit_count")}
        dev_metric[policy]["overall"] = combined | {"valid_rate": combined["valid_count"] / combined["sample_count"], "annotation_rate": combined["annotation_canonical_count"] / combined["sample_count"]}
    selected = select_policy({policy: dev_metric[policy]["overall"] for policy in POLICIES})
    frozen = {"schema_version": "current_kb_exact_frozen_ranking_policy_v2", "selected_policy": selected, "feature_order": POLICY_FEATURES[selected], "selection_dataset": "BehaviorV5Dev538", "final2000_used_for_selection": False, "ambiguity_policy": "v1-equivalent", "entries_sha256": sha256(ENTRIES), "aliases_sha256": sha256(ALIASES)}
    write_json(OUT / "config/frozen_ranking_policy_v2.json", frozen)
    frozen["policy_sha256"] = sha256(OUT / "config/frozen_ranking_policy_v2.json")
    # One Final2000 execution only after frozen policy exists.
    final_metric = {}
    for direction in ("bo2zh", "zh2bo"):
        rows = [retrieve_row(source_only(row), index, selected) for row in final[direction]]
        write_jsonl(OUT / "retrieval" / f"{direction}_frozen_{selected}.jsonl", rows)
        final_metric[direction] = metrics(rows, final[direction])
    write_json(OUT / "evaluation/dev_policy_metrics.json", dev_metric)
    write_json(OUT / "evaluation/final2000_frozen_policy_metrics.json", final_metric)
    write_json(OUT / "manifests/policy_selection_manifest.json", frozen)
    write_json(OUT / "manifests/run_manifest.json", {"translation_inference": {"started": False, "model_imports": []}, "final2000_policy_runs": 1, "selected_policy": selected})


if __name__ == "__main__":
    main()
