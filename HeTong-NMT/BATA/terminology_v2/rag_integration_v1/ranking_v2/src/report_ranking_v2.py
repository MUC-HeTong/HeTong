#!/usr/bin/env python3
"""Generate ranking-v2 diagnostics/reports from frozen retrieval artifacts only."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT.parent.parent
FINAL = PROJECT / "backbone_migration_hy2_ckpt400k/step32_bidirectional_final_eval_v1/data"
V1 = ROOT
OUT = ROOT / "ranking_v2"
FEATURES = ("normalized_length_desc", "raw_length_desc", "coverage_desc", "source_count_desc", "source_pair_count_desc", "start_asc", "surface_asc", "term_id_asc")

def rows(path): return [json.loads(x) for x in path.read_text(encoding="utf8").splitlines() if x]
def write_json(path, value): path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf8")
def write_jsonl(path, value): path.parent.mkdir(parents=True, exist_ok=True); path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True)+"\n" for x in value), encoding="utf8")
def canonical_bo(value): return " ".join(value.strip().rstrip("།").split())

def relation(a, b):
    if a[0] <= b[0] and b[1] <= a[1]: return "GOLD_CONTAINED_IN_SELECTED_LONGER_SPAN"
    if b[0] <= a[0] and a[1] <= b[1]: return "SELECTED_CONTAINED_IN_GOLD_SPAN"
    if a[0] < b[1] and b[0] < a[1]: return "PARTIAL_OVERLAP"
    return "DISJOINT"

def main():
    v1_errors = {(r["direction"], r["record_id"]): r for r in rows(V1 / "evaluation/retrieval_errors.jsonl") if r["failure_taxonomy"] == "OVERLAP_SELECTION_ERROR"}
    finals = {d: {str(r["record_id"]): r for r in rows(FINAL / f"{d}_final2000.jsonl")} for d in ("bo2zh", "zh2bo")}
    v1 = {d: {str(r["record_id"]): r for r in rows(V1 / "retrieval" / f"{d}_retrieval.jsonl")} for d in ("bo2zh", "zh2bo")}
    r5 = {d: {str(r["record_id"]): r for r in rows(OUT / "retrieval" / f"{d}_frozen_R5.jsonl")} for d in ("bo2zh", "zh2bo")}
    diagnostics=[]; counts=Counter(); deciding=Counter(); relations=Counter()
    for (d, record_id), error in sorted(v1_errors.items()):
        gold=finals[d][record_id]; old=v1[d][record_id]; trace=r5[d][record_id]
        canonical_target = gold["zh_term"] if d == "bo2zh" else canonical_bo(gold["bo_term"])
        gold_source = gold["bo_term"] if d == "bo2zh" else gold["zh_term"]
        candidates=[x for x in trace["all_matched_spans"] if x["surface"] == gold_source and x["canonical_target_term"] == canonical_target]
        selected=[x for x in trace["all_matched_spans"] if x["surface"] == old["matched_source_surface"] and x["target_term"] == old["target_term"] and x["start"] == old["matched_span"]["start"]]
        category="A_GOLD_PRESENT_RANKED_BELOW_OTHER" if candidates else "B_GOLD_ABSENT_FROM_CANDIDATE_SET"
        counts[category]+=1
        geometry=None; first=None
        if candidates and selected:
            gc=min(candidates,key=lambda x: tuple(x["r1_rank_key"])); sc=selected[0]
            geometry=relation((sc["start"],sc["end"]),(gc["start"],gc["end"])); relations[geometry]+=1
            for name, win, gold_value in zip(FEATURES, sc["r1_rank_key"], gc["r1_rank_key"]):
                if win != gold_value: first=name; deciding[name]+=1; break
        diagnostics.append({"record_id":record_id,"direction":d,"source":gold["source"],"gold_source_term":gold_source,"gold_target_term":gold["target_term"],"v1_selected_source_surface":old["matched_source_surface"],"v1_selected_target":old["target_term"],"all_matched_spans":trace["all_matched_spans"],"gold_candidate_status":category,"span_geometry":geometry,"first_deciding_r1_feature":first})
    write_jsonl(OUT/"diagnostics/v1_overlap_selection_diagnostics.jsonl",diagnostics)
    diag_summary={"overlap_error_count":len(diagnostics),"gold_candidate_presence":dict(counts),"geometry":dict(relations),"first_deciding_r1_feature":dict(deciding)}
    write_json(OUT/"diagnostics/v1_overlap_diagnostic_summary.json",diag_summary)
    v1metrics=json.loads((V1/"evaluation/retrieval_metrics.json").read_text(encoding="utf8"))
    r5metrics=json.loads((OUT/"evaluation/final2000_frozen_policy_metrics.json").read_text(encoding="utf8"))
    comparison={"v1":v1metrics,"v2_frozen_R5":r5metrics,"diagnostics":diag_summary,"note":"R5 was selected only on BehaviorV5Dev538; Final2000 figures are one frozen-policy evaluation."}
    write_json(OUT/"evaluation/v1_vs_frozen_r5_comparison.json",comparison)
    devmetrics=json.loads((OUT/"evaluation/dev_policy_metrics.json").read_text(encoding="utf8"))
    report=["# Ranking v2 Audit","","## Selection boundary","R5 was selected only from leakage-audited BehaviorV5Dev538. Final2000 did not select its feature order.","","## Dev Policy Comparison","| Policy | Valid retrieval | Canonical annotation alignment | Non-HIT |","| --- | ---: | ---: | ---: |"]
    for policy, value in sorted(devmetrics.items()):
        m=value["overall"]; report.append(f"| {policy} | {m['valid_rate']:.2%} | {m['annotation_rate']:.2%} | {m['nonhit_count']} |")
    report += ["", "Selection maximizes valid retrieval, then canonical annotation alignment, then minimizes non-HIT outcomes, then uses policy name. R5 wins on Dev486-independent BehaviorV5Dev538.", "", "## Overlap diagnosis",f"- v1 OVERLAP_SELECTION_ERROR rows: {len(diagnostics)}",f"- Gold candidate present / ranked below: {counts['A_GOLD_PRESENT_RANKED_BELOW_OTHER']}",f"- Gold candidate absent: {counts['B_GOLD_ABSENT_FROM_CANDIDATE_SET']}",f"- First R1 deciding features: {dict(deciding)}",f"- Span geometry: {dict(relations)}","","## Frozen Final2000 R5","| Direction | Valid retrieval | Exact annotation alignment | Canonical annotation alignment | Alternative valid | Non-HIT |","| --- | ---: | ---: | ---: | ---: | ---: |"]
    for d in ("bo2zh","zh2bo"):
        m=r5metrics[d]; report.append(f"| {d} | {m['valid_rate']:.2%} | {m['annotation_exact_count']/m['sample_count']:.2%} | {m['annotation_rate']:.2%} | {m['taxonomy'].get('ALTERNATIVE_VALID_TERM',0)} | {m['nonhit_count']} |")
    v1overall=v1metrics["overall"]
    r5combined={key:sum(r5metrics[d][key] for d in ("bo2zh","zh2bo")) for key in ("sample_count","annotation_exact_count","annotation_canonical_count","valid_count","nonhit_count")}
    v1overlap=sum(v1metrics["directions"][d]["failure_taxonomy_counts"].get("OVERLAP_SELECTION_ERROR",0) for d in ("bo2zh","zh2bo"))
    report += ["", "## v1 vs Frozen R5", "| Metric | v1 | Frozen R5 |", "| --- | ---: | ---: |", f"| Valid terminology retrieval | {v1overall['status_counts']['HIT']/v1overall['sample_count']:.2%} | {r5combined['valid_count']/r5combined['sample_count']:.2%} |", f"| Exact annotated-pair agreement | {v1overall['exact_gold_target_agreement_count']/v1overall['sample_count']:.2%} | {r5combined['annotation_exact_count']/r5combined['sample_count']:.2%} |", f"| Canonical annotated-pair agreement | {v1overall['overall_canonical_gold_agreement_rate']:.2%} | {r5combined['annotation_canonical_count']/r5combined['sample_count']:.2%} |", f"| Alternative valid term | {v1overlap} | {sum(r5metrics[d]['taxonomy'].get('ALTERNATIVE_VALID_TERM',0) for d in ('bo2zh','zh2bo'))} |", f"| Ambiguous + blocked | {v1overall['status_counts']['AMBIGUOUS_MATCH'] + v1overall['status_counts']['BIDIRECTIONAL_CONFLICT_BLOCKED']} | {r5combined['nonhit_count']} |", "", "`ALTERNATIVE_VALID_TERM` is structural current-KB validity of the selected source span, not human semantic validation."]
    (OUT/"reports").mkdir(parents=True,exist_ok=True); (OUT/"reports/RANKING_V2_AUDIT.md").write_text("\n".join(report)+"\n",encoding="utf8")

if __name__=="__main__": main()
