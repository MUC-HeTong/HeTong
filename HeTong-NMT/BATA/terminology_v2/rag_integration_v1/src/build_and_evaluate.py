#!/usr/bin/env python3
"""Build current-KB exact indexes and run Final2000 retrieval-only evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ROOT = Path(os.environ.get('BATA_TERMINOLOGY_ROOT', ROOT)).expanduser().resolve()
PROJECT = Path(os.environ.get('BATA_PROJECT_ROOT', ROOT.parent)).expanduser().resolve()
sys.path.insert(0, str(HERE))
from current_kb_exact_retriever import audit_index, build_index, retrieve, to_retriever_input


OUT = ROOT / "rag_integration_v1"
KB_ENTRIES = Path(os.environ.get('BATA_TERMINOLOGY_ENTRIES', ROOT / "terminology_entries.jsonl"))
KB_ALIASES = Path(os.environ.get('BATA_TERMINOLOGY_ALIASES', ROOT / "terminology_aliases.jsonl"))
POLICY = OUT / "config/retrieval_policy_v1.json"
FINAL_INPUT = Path(os.environ.get("BATA_FINAL_DATA_DIR", PROJECT / "data/final2000"))
DIRS = ("bo2zh", "zh2bo")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonicalize_bo(value: str) -> str:
    return " ".join(value.strip().rstrip("།").split())


def _gold_source_in_index(gold_source: str, direction: str, index: dict[str, Any]) -> bool:
    return gold_source in index[direction]


def enrich_gold_for_evaluation(rows: list[dict[str, Any]], entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Derive checker metadata from current KB only after retrieval output is frozen."""
    by_pair = {(entry["zh_term"], entry["bo_canonical"]): entry for entry in entries}
    output = []
    for row in rows:
        entry = by_pair.get((row["zh_term"], canonicalize_bo(row["bo_term"])))
        if entry is None:
            raise ValueError(f"current KB has no evaluation metadata pair for {row['record_id']}")
        canonical_target = entry["zh_term"] if row["direction"] == "bo2zh" else entry["bo_canonical"]
        surfaces = [entry["zh_term"]] if row["direction"] == "bo2zh" else sorted(set(entry["bo_surface_forms"]) | {entry["bo_canonical"]})
        output.append(row | {
            "canonical_target_term": canonical_target,
            "surface_target_terms": surfaces,
            "terminology_id": entry["term_id"],
        })
    return output


def _taxonomy(result: dict[str, Any], gold: dict[str, Any], index: dict[str, Any]) -> tuple[str, bool, bool]:
    exact = result["retrieval_status"] == "HIT" and result["target_term"] == gold["target_term"]
    canonical = result["retrieval_status"] == "HIT" and (
        result["canonical_target_term"] == gold["canonical_target_term"]
        or result["target_term"] in set(gold["surface_target_terms"])
    )
    gold_source = gold["source_term"]
    if result["retrieval_status"] == "HIT":
        if exact:
            return "CORRECT_HIT", exact, canonical
        if canonical:
            return "CANONICAL_ONLY_AGREEMENT", exact, canonical
        if _gold_source_in_index(gold_source, gold["direction"], index):
            return "OVERLAP_SELECTION_ERROR", exact, canonical
        return "WRONG_PAIR", exact, canonical
    if result["retrieval_status"] == "AMBIGUOUS_MATCH":
        return ("AMBIGUOUS_GOLD_SURFACE" if result["matched_source_surface"] == gold_source else "AMBIGUOUS_OTHER_SURFACE"), exact, canonical
    if result["retrieval_status"] == "BIDIRECTIONAL_CONFLICT_BLOCKED":
        return "BIDIRECTIONAL_CONFLICT_BLOCKED", exact, canonical
    if gold_source not in gold["source"]:
        return "GOLD_SOURCE_NOT_IN_SENTENCE", exact, canonical
    if not _gold_source_in_index(gold_source, gold["direction"], index):
        normalized = canonicalize_bo(gold_source) if gold["direction"] == "bo2zh" else gold_source
        if normalized in index[gold["direction"]]:
            return "NORMALIZATION_OR_ALIAS_MISS", exact, canonical
        return "KB_MISSING", exact, canonical
    return "NO_HIT_OTHER", exact, canonical


def evaluate(direction: str, results: list[dict[str, Any]], gold_rows: list[dict[str, Any]], index: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if [row["record_id"] for row in results] != [str(row["record_id"]) for row in gold_rows]:
        raise ValueError(f"{direction}: retrieval/gold record ID order mismatch")
    status = Counter()
    taxonomy = Counter()
    hit_count = exact_count = canonical_count = 0
    errors: list[dict[str, Any]] = []
    for result, gold in zip(results, gold_rows):
        label, exact, canonical = _taxonomy(result, gold, index)
        status[result["retrieval_status"]] += 1
        taxonomy[label] += 1
        hit_count += result["retrieval_status"] == "HIT"
        exact_count += exact
        canonical_count += canonical
        errors.append({
            "schema_version": "current_kb_retrieval_error_v1",
            "record_id": result["record_id"], "direction": direction,
            "retrieval_status": result["retrieval_status"], "failure_taxonomy": label,
            "matched_source_surface": result["matched_source_surface"],
            "match_type": result["match_type"], "candidate_entry_count": result["candidate_entry_count"],
            "distinct_target_count": result["distinct_target_count"],
            "would_be_unique_without_bidirectional_block": result["would_be_unique_without_bidirectional_block"],
            "gold_source_in_sentence": gold["source_term"] in gold["source"],
            "gold_source_in_current_index": _gold_source_in_index(gold["source_term"], direction, index),
            "exact_gold_target_agreement": exact, "canonical_gold_target_agreement": canonical,
        })
    count = len(results)
    return {
        "sample_count": count,
        "status_counts": dict(sorted(status.items())),
        "status_rates": {name: status[name] / count for name in ("HIT", "NO_HIT", "AMBIGUOUS_MATCH", "BIDIRECTIONAL_CONFLICT_BLOCKED")},
        "hit_count": hit_count,
        "exact_gold_target_agreement_count": exact_count,
        "exact_gold_target_agreement_rate_among_hits": exact_count / hit_count if hit_count else None,
        "overall_end_to_end_retrieval_gold_agreement_count": exact_count,
        "overall_end_to_end_retrieval_gold_agreement_rate": exact_count / count,
        "canonical_gold_agreement_count": canonical_count,
        "canonical_gold_agreement_rate_among_hits": canonical_count / hit_count if hit_count else None,
        "overall_canonical_gold_agreement_rate": canonical_count / count,
        "failure_taxonomy_counts": dict(sorted(taxonomy.items())),
    }, errors


def _format_metric_row(direction: str, metric: dict[str, Any]) -> str:
    rates = metric["status_rates"]
    return "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
        direction, metric["sample_count"], f"{rates['HIT']:.2%}", f"{rates['NO_HIT']:.2%}",
        f"{rates['AMBIGUOUS_MATCH']:.2%}", f"{rates['BIDIRECTIONAL_CONFLICT_BLOCKED']:.2%}",
        f"{metric['exact_gold_target_agreement_rate_among_hits']:.2%}",
        f"{metric['overall_end_to_end_retrieval_gold_agreement_rate']:.2%}",
        f"{metric['canonical_gold_agreement_rate_among_hits']:.2%}",
        f"{metric['overall_canonical_gold_agreement_rate']:.2%}",
    )


def write_reports(index_audit: dict[str, Any], metrics: dict[str, Any]) -> None:
    report_dir = OUT / "reports"
    rows = [_format_metric_row(direction, metrics["directions"][direction]) for direction in DIRS]
    overlap_failures = sum(metrics["directions"][direction]["failure_taxonomy_counts"].get("OVERLAP_SELECTION_ERROR", 0) for direction in DIRS)
    canonical_wrong = metrics["overall"]["sample_count"] - metrics["overall"]["canonical_gold_agreement_count"]
    gate = "HOLD: do not start full Retrieved-M3/LoRA/Correction translation inference"
    index_rows = [
        "| {} | {} | {} | {} | {} | {} | {} |".format(
            direction, data["unique_source_surfaces"], data["source_surfaces_with_exactly_one_target"],
            data["source_surfaces_with_multiple_targets"], data["canonical_source_ambiguities"],
            data["alias_source_ambiguities"], data["source_surfaces_with_bidirectional_conflict"],
        ) for direction, data in index_audit.items()
    ]
    (report_dir / "INTEGRATION_PLAN_AND_RETRIEVAL_AUDIT.md").write_text("\n".join([
        "# Current-KB RAG Integration and Retrieval Audit", "",
        "## Historical M3 v2 Adaptation", "",
        "Reused: direction-specific raw-source exact span matching, all-candidate collection, deterministic ranking, single selection, no-term output on unavailable constraints, and provenance recording. Rewritten: current `TERM_*` resource loading, alias index, source-surface uniqueness decision, bidirectional block, current canonical metadata, and gold-isolated Final2000 evaluation.", "",
        "## Policy", "",
        "The retriever reads only `record_id`, `direction`, and `source`. BO->ZH scans canonical Tibetan and observed Tibetan surfaces; ZH->BO scans exact Chinese terms. A surface returns only when its current-KB distinct target set has cardinality one and none of its candidates has `BIDIRECTIONAL_CONFLICT`. Other-direction conflict provenance is retained, not globally filtered. Structural uniqueness is not human semantic verification.", "",
        "## Index Audit", "",
        "| Direction | Source surfaces | One target | Multiple targets | Canonical ambiguity | Alias ambiguity | Bidirectional surfaces |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |", *index_rows, "",
        "## Final2000 Retrieval-only Results", "",
        "| Direction | Samples | HIT | NO_HIT | AMBIGUOUS | BIDIRECTIONAL_BLOCKED | Exact among HIT | Exact overall | Canonical among HIT | Canonical overall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |", *rows, "",
        "## Gate 5 Recommendation", "",
        f"**{gate}.** Although structural HIT coverage is {metrics['overall']['status_counts']['HIT']}/{metrics['overall']['sample_count']} ({metrics['overall']['status_counts']['HIT'] / metrics['overall']['sample_count']:.2%}), only {metrics['overall']['canonical_gold_agreement_count']}/{metrics['overall']['sample_count']} ({metrics['overall']['overall_canonical_gold_agreement_rate']:.2%}) agree with the gold canonical target. `{overlap_failures}` rows are classified as `OVERLAP_SELECTION_ERROR`. A full translation experiment would conflate this unresolved retrieval ranking error with model utilization; audit/rank remediation is required first.", "",
        "No translation model, adapter, checkpoint, or unified evaluator was invoked. The Gate 5 recommendation is recorded after reviewing these frozen metrics in `RETRIEVAL_EVALUATION.md`.", "",
    ]), encoding="utf-8")
    (report_dir / "RETRIEVAL_EVALUATION.md").write_text("\n".join([
        "# Retrieval-only Evaluation", "",
        "This evaluation joins frozen retrieval outputs to Final2000 gold metadata only after output serialization. Exact agreement is retrieved target equals gold `target_term`. Canonical agreement accepts retrieved canonical target equality with gold canonical target or membership in the gold registered target surfaces.", "",
        "| Direction | Samples | HIT | NO_HIT | AMBIGUOUS | BIDIRECTIONAL_BLOCKED | Exact among HIT | Exact overall | Canonical among HIT | Canonical overall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |", *rows, "",
        "## Gate 4", "", "Retrieval-only outputs, index audit, error rows, source-only inputs, and manifests have been produced. No end-to-end translation experiment was started.", "",
        "## Gate 5 Recommendation", "",
        f"**{gate}.** Canonical agreement is {metrics['overall']['overall_canonical_gold_agreement_rate']:.2%} overall, with `{overlap_failures}` `OVERLAP_SELECTION_ERROR` rows and `{canonical_wrong}` rows lacking canonical gold agreement. Diagnose/revise ranking before a model experiment so retrieval and utilization failures remain interpretable.", "",
        "## Alias Ambiguity Impact (BO->ZH)", "",
        f"Observed alias-ambiguous output count: `{metrics['alias_ambiguity_impact_bo2zh']['observed_alias_ambiguous_output_count']}`. Canonical-only counterfactual statuses for those outputs: `{metrics['alias_ambiguity_impact_bo2zh']['canonical_only_statuses_for_alias_ambiguous_outputs']}`. Alias-enabled minus canonical-only HIT count: `{metrics['alias_ambiguity_impact_bo2zh']['observed_alias_enabled_hit_minus_canonical_only_hit']}`. This is source-only and does not establish semantic correctness.", "",
    ]), encoding="utf-8")
    taxonomy_rows = []
    for direction in DIRS:
        for label, count in metrics["directions"][direction]["failure_taxonomy_counts"].items():
            taxonomy_rows.append(f"| {direction} | {label} | {count} |")
    (report_dir / "ERROR_ANALYSIS.md").write_text("\n".join([
        "# Retrieval Error Analysis", "",
        "Taxonomy labels describe structural retrieval behavior under the current KB. They do not establish semantic correctness. `OVERLAP_SELECTION_ERROR` means the gold source surface exists in the current direction index but another eligible span ranked first; `KB_MISSING` means the gold source surface is absent from the raw exact index.", "",
        "| Direction | Taxonomy | Count |", "| --- | --- | ---: |", *taxonomy_rows, "",
    ]), encoding="utf-8")


def main() -> None:
    for directory in (OUT / "indexes", OUT / "retrieval", OUT / "evaluation", OUT / "reports", OUT / "manifests"):
        directory.mkdir(parents=True, exist_ok=True)
    entries, aliases = read_jsonl(KB_ENTRIES), read_jsonl(KB_ALIASES)
    index = build_index(entries, aliases)
    index_audit = audit_index(index)
    index_path = OUT / "indexes/current_kb_exact_index_v1.json"
    write_json(index_path, index)
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    index_manifest = {
        "schema_version": "current_kb_exact_index_manifest_v1",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "entries_path": str(KB_ENTRIES), "entries_sha256": sha256(KB_ENTRIES), "entries_count": len(entries),
        "aliases_path": str(KB_ALIASES), "aliases_sha256": sha256(KB_ALIASES), "aliases_count": len(aliases),
        "policy_path": str(POLICY), "policy_sha256": sha256(POLICY), "policy_schema_version": policy["schema_version"],
        "builder_path": str(Path(__file__)), "builder_sha256": sha256(Path(__file__)),
        "retriever_path": str(HERE / "current_kb_exact_retriever.py"), "retriever_sha256": sha256(HERE / "current_kb_exact_retriever.py"),
        "index_path": str(index_path), "index_sha256": sha256(index_path), "index_audit": index_audit,
        "serialization": "UTF-8 JSON, sorted object keys; no pickle or external index dependency",
    }
    write_json(OUT / "manifests/index_manifest.json", index_manifest)

    all_metrics: dict[str, Any] = {"schema_version": "current_kb_retrieval_metrics_v1", "directions": {}}
    all_errors: list[dict[str, Any]] = []
    retrieval_inputs: dict[str, list[dict[str, Any]]] = {}
    retrieval_results: dict[str, list[dict[str, Any]]] = {}
    retrieval_manifest: dict[str, Any] = {
        "schema_version": "current_kb_retrieval_manifest_v1", "index_manifest_sha256": sha256(OUT / "manifests/index_manifest.json"),
        "retriever_input_fields": ["record_id", "direction", "source"],
        "gold_fields_excluded_from_retrieval": ["reference", "source_term", "target_term", "canonical_target_term", "surface_target_terms", "terminology_id"],
        "retrieval_output_note": "target_term, canonical_target_term, surface_target_terms, and terminology_id are retriever-produced fields. Field-name overlap with gold metadata is not evidence of gold leakage.",
        "directions": {}, "translation_inference": {"started": False, "model_imports": []},
    }
    for direction in DIRS:
        full_rows = read_jsonl(FINAL_INPUT / f"{direction}_final2000.jsonl")
        if len(full_rows) != 2000 or any(row["direction"] != direction for row in full_rows):
            raise ValueError(f"{direction}: expected 2,000 direction-matched Final2000 records")
        inputs = [to_retriever_input(row) for row in full_rows]
        retrieval_inputs[direction] = inputs
        input_path = OUT / "retrieval" / f"{direction}_retriever_inputs.jsonl"
        write_jsonl(input_path, inputs)
        if any(set(row) != {"record_id", "direction", "source"} for row in read_jsonl(input_path)):
            raise ValueError(f"{direction}: retrieval input gold-isolation check failed")
        results = [retrieve(row, index) for row in inputs]
        retrieval_results[direction] = results
        result_path = OUT / "retrieval" / f"{direction}_retrieval.jsonl"
        write_jsonl(result_path, results)
        before_eval_hash = sha256(result_path)
        gold_rows = enrich_gold_for_evaluation(full_rows, entries)
        metrics, errors = evaluate(direction, read_jsonl(result_path), gold_rows, index)
        if before_eval_hash != sha256(result_path):
            raise ValueError(f"{direction}: evaluation modified retrieval output")
        all_metrics["directions"][direction] = metrics
        all_errors.extend(errors)
        retrieval_manifest["directions"][direction] = {
            "final2000_input_path": str(FINAL_INPUT / f"{direction}_final2000.jsonl"),
            "final2000_input_sha256": sha256(FINAL_INPUT / f"{direction}_final2000.jsonl"),
            "retriever_input_path": str(input_path), "retriever_input_sha256": sha256(input_path),
            "retrieval_output_path": str(result_path), "retrieval_output_sha256": sha256(result_path),
            "record_count": len(results), "unique_record_id_count": len({row["record_id"] for row in results}),
            "output_contains_reference": any("reference" in row for row in results),
            "output_contains_gold_source_term": any("source_term" in row for row in results),
        }
    total = sum(all_metrics["directions"][direction]["sample_count"] for direction in DIRS)
    all_metrics["overall"] = {
        "sample_count": total,
        "status_counts": {status: sum(all_metrics["directions"][d]["status_counts"].get(status, 0) for d in DIRS) for status in ("HIT", "NO_HIT", "AMBIGUOUS_MATCH", "BIDIRECTIONAL_CONFLICT_BLOCKED")},
        "exact_gold_target_agreement_count": sum(all_metrics["directions"][d]["exact_gold_target_agreement_count"] for d in DIRS),
        "canonical_gold_agreement_count": sum(all_metrics["directions"][d]["canonical_gold_agreement_count"] for d in DIRS),
    }
    all_metrics["overall"]["overall_end_to_end_retrieval_gold_agreement_rate"] = all_metrics["overall"]["exact_gold_target_agreement_count"] / total
    all_metrics["overall"]["overall_canonical_gold_agreement_rate"] = all_metrics["overall"]["canonical_gold_agreement_count"] / total
    canonical_only_index = build_index(entries, include_observed_surfaces=False)
    canonical_only_results = [retrieve(row, canonical_only_index) for row in retrieval_inputs["bo2zh"]]
    alias_ambiguous = [
        (observed, canonical)
        for observed, canonical in zip(retrieval_results["bo2zh"], canonical_only_results)
        if observed["retrieval_status"] == "AMBIGUOUS_MATCH" and observed["match_type"] == "alias"
    ]
    observed_alias_enabled_hits = sum(row["retrieval_status"] == "HIT" for row in retrieval_results["bo2zh"])
    canonical_only_hits = sum(row["retrieval_status"] == "HIT" for row in canonical_only_results)
    all_metrics["alias_ambiguity_impact_bo2zh"] = {
        "method": "source-only canonical-only counterfactual; no gold fields or translation model used",
        "observed_alias_ambiguous_output_count": len(alias_ambiguous),
        "canonical_only_statuses_for_alias_ambiguous_outputs": dict(sorted(Counter(canonical["retrieval_status"] for _, canonical in alias_ambiguous).items())),
        "alias_ambiguous_outputs_that_canonical_only_would_hit": sum(canonical["retrieval_status"] == "HIT" for _, canonical in alias_ambiguous),
        "observed_alias_enabled_hit_count": observed_alias_enabled_hits,
        "canonical_only_hit_count": canonical_only_hits,
        "observed_alias_enabled_hit_minus_canonical_only_hit": observed_alias_enabled_hits - canonical_only_hits,
    }
    write_json(OUT / "evaluation/retrieval_metrics.json", all_metrics)
    write_jsonl(OUT / "evaluation/retrieval_errors.jsonl", all_errors)
    retrieval_manifest["evaluation_metrics_sha256"] = sha256(OUT / "evaluation/retrieval_metrics.json")
    retrieval_manifest["error_rows_sha256"] = sha256(OUT / "evaluation/retrieval_errors.jsonl")
    write_json(OUT / "manifests/retrieval_manifest.json", retrieval_manifest)
    write_reports(index_audit, all_metrics)
    print(json.dumps({"index_audit": index_audit, "metrics": all_metrics, "translation_inference_started": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
