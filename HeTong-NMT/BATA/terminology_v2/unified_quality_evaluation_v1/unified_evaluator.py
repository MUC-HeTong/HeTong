#!/usr/bin/env python3
"""Read-only unified quality evaluator using the direct evaluation protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Any, Optional

import jieba
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU, CHRF


TIBETAN_SEPARATORS = re.compile(r"[\s་།༎༏༐༑༔]+")
CJK_PATTERN = re.compile(r"[一-鿿㐀-䶿豈-﫿]")
TIBETAN_PATTERN = re.compile(r"[ༀ-࿿]")
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent


def clean(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", str(text)).strip().split())


def tibetan_tokens(text: str) -> list[str]:
    return [token for token in TIBETAN_SEPARATORS.split(clean(text)) if token]


def detect_lang(text: str) -> str:
    if TIBETAN_PATTERN.search(text):
        return "tibetan"
    if CJK_PATTERN.search(text):
        return "chinese"
    return "other"


def tokenize(text: str, lang: Optional[str] = None) -> list[str]:
    language = lang or detect_lang(text)
    if language == "tibetan":
        return tibetan_tokens(text)
    if language == "chinese":
        return list(jieba.cut(clean(text)))
    return clean(text).split()


class LangAwareTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return tokenize(text)


def validate_rows(rows: list[dict[str, Any]], expected_count: int | None = None) -> dict[str, Any]:
    record_ids = [str(row["record_id"]) for row in rows]
    duplicate_count = len(record_ids) - len(set(record_ids))
    if duplicate_count:
        raise ValueError(f"duplicate record_id count: {duplicate_count}")
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} rows, got {len(rows)}")
    return {
        "num_samples": len(rows),
        "record_id_count": len(set(record_ids)),
        "duplicate_count": duplicate_count,
        "missing_count": 0 if expected_count is None else max(0, expected_count - len(rows)),
    }


def validate_id_alignment(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> None:
    left_ids = [str(row["record_id"]) for row in left]
    right_ids = [str(row["record_id"]) for row in right]
    if left_ids != right_ids:
        raise ValueError("record_id order does not match")


def normalize_condition_rows(source_rows: list[dict[str, Any]], condition: str) -> list[dict[str, Any]]:
    output = []
    for row in source_rows:
        output.append({
            "record_id": row["record_id"],
            "direction": row["direction"],
            "reference": row["reference"],
            "prediction": row[f"{condition}_prediction"],
            "strict_term_pass": row[f"{condition}_term_pass"],
            "canonical_term_pass": row[f"{condition}_canonical_term_pass"],
        })
    return output


@lru_cache(maxsize=None)
def score_pair(reference: str, prediction: str) -> tuple[float, float, float]:
    rouge = rouge_scorer.RougeScorer(["rougeL"], tokenizer=LangAwareTokenizer())
    reference, prediction = clean(reference), clean(prediction)
    language = detect_lang(reference)
    bleu_ref = " ".join(tokenize(reference, language))
    bleu_hyp = " ".join(tokenize(prediction, language))
    return (
        BLEU(tokenize="none", effective_order=True).sentence_score(bleu_hyp, [bleu_ref]).score,
        CHRF(word_order=2).sentence_score(prediction, [reference]).score,
        rouge.score(reference, prediction)["rougeL"].fmeasure,
    )


def evaluate_rows(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("cannot evaluate empty rows")
    scores = [score_pair(row["reference"], row["prediction"]) for row in rows]
    strict = [bool(row["strict_term_pass"]) for row in rows]
    canonical = [bool(row["canonical_term_pass"]) for row in rows]
    return {
        "sample_count": len(rows),
        "BLEU": mean(score[0] for score in scores),
        "chrF++": mean(score[1] for score in scores),
        "ROUGE-L": mean(score[2] for score in scores),
        "strict_term_success": sum(strict),
        "strict_term_accuracy": sum(strict) / len(rows),
        "canonical_term_success": sum(canonical),
        "canonical_term_accuracy": sum(canonical) / len(rows),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required = {"record_id", "direction", "reference", "prediction", "strict_term_pass", "canonical_term_pass"}
    if not source_rows:
        raise ValueError("source prediction JSONL is empty")
    missing = required - set(source_rows[0])
    if missing:
        raise ValueError(f"source is missing fields: {sorted(missing)}")
    return [{key: row[key] for key in required} for row in source_rows]


def load_source(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    before = sha256(path)
    rows = normalize_rows(read_jsonl(path))
    after = sha256(path)
    return rows, {"path": str(path), "sha256_before": before, "sha256_after": after, "sha256_unchanged": before == after}


def old_metric_source(experiment: str, direction: str, method: str) -> dict[str, Any] | None:
    mappings = {
        "System Ablation": (ROOT / "stage4_system_ablation_v1/stage4_system_ablation_quality_report.json", ("directions", direction, "methods", method, "all_final2000")),
        "Ablation 1": (ROOT / "ablation_lora_correction_v1/lora_only_vs_correction_report.json", ("directions", direction, method)),
        "Ablation 2": (ROOT / "ablation_direction_vs_joint_lora_v1/direction_vs_joint_lora_report.json", ("directions", direction, method)),
        "Ablation 3": (ROOT / "ablation_lora_data_composition_v1/lora_data_composition_report.json", ("directions", direction, "metrics", method)),
    }
    if experiment not in mappings:
        return None
    path, keys = mappings[experiment]
    if not path.is_file() or path.stat().st_size == 0:
        return None
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    for key in keys:
        value = value.get(key) if isinstance(value, dict) else None
        if value is None:
            return None
    return value


def unavailable(path: Path, reason: str) -> dict[str, Any]:
    return {"status": "UNAVAILABLE_SOURCE", "source_path": str(path), "reason": reason}


def registry() -> list[dict[str, Any]]:
    stage4 = ROOT / "stage4_system_ablation_v1"
    composition = ROOT / "ablation_lora_data_composition_v1/predictions"
    stage6 = ROOT / "stage6_canonical_lora_first"
    missing_stage3 = ROOT / "stage3_evaluation_vllm_v1"
    entries: list[dict[str, Any]] = []
    for direction in ("bo2zh", "zh2bo"):
        entries.extend([
            {"experiment": "Main Experiment", "method": "Base", "direction": direction, "missing": missing_stage3 / f"{direction}_predictions.jsonl", "reason": "Complete Base prediction field is unavailable: current Stage3 vLLM source JSONL is empty."},
            {"experiment": "Main Experiment", "method": "M3", "direction": direction, "path": stage4 / "always_on_m3" / f"{direction}_predictions.jsonl"},
            {"experiment": "Main Experiment", "method": "Direction-specific LoRA", "direction": direction, "path": stage4 / "always_on_lora" / f"{direction}_predictions.jsonl"},
            {"experiment": "Main Experiment", "method": "Joint LoRA", "direction": direction, "missing": missing_stage3 / f"{direction}_predictions.jsonl", "reason": "Complete Joint-LoRA prediction field is unavailable: current Stage3 vLLM source JSONL is empty."},
            {"experiment": "Main Experiment", "method": "Ours cascade", "direction": direction, "missing": missing_stage3 / f"{direction}_predictions.jsonl", "reason": "Original cascade outputs cannot be reconstructed without complete Base/M3/LoRA source fields."},
            {"experiment": "System Ablation", "method": "ours_cascade", "direction": direction, "missing": missing_stage3 / f"{direction}_predictions.jsonl", "reason": "Stage4 Ours cascade is derived from the empty Stage3 vLLM source JSONL."},
            {"experiment": "System Ablation", "method": "always_on_m3", "direction": direction, "path": stage4 / "always_on_m3" / f"{direction}_predictions.jsonl"},
            {"experiment": "System Ablation", "method": "always_on_lora", "direction": direction, "path": stage4 / "always_on_lora" / f"{direction}_predictions.jsonl"},
            {"experiment": "System Ablation", "method": "no_routing", "direction": direction, "path": stage4 / "no_routing" / f"{direction}_predictions.jsonl"},
            {"experiment": "Ablation 1", "method": "lora_only", "direction": direction, "path": stage4 / "always_on_lora" / f"{direction}_predictions.jsonl"},
            {"experiment": "Ablation 1", "method": "lora_plus_canonical_correction", "direction": direction, "path": stage6 / f"{direction}_canonical_lora_first_predictions.jsonl"},
            {"experiment": "Ablation 2", "method": "direction_lora", "direction": direction, "path": stage4 / "always_on_lora" / f"{direction}_predictions.jsonl"},
            {"experiment": "Ablation 2", "method": "joint_lora", "direction": direction, "missing": missing_stage3 / f"{direction}_predictions.jsonl", "reason": "Complete Joint-LoRA prediction field is unavailable: current Stage3 vLLM source JSONL is empty."},
            {"experiment": "Ablation 3", "method": "full_dataset", "direction": direction, "path": stage4 / "always_on_lora" / f"{direction}_predictions.jsonl"},
            {"experiment": "Ablation 3", "method": "rescue_only", "direction": direction, "path": composition / "rescue_only" / f"{direction}_predictions.jsonl"},
            {"experiment": "Ablation 3", "method": "rescue_preservation", "direction": direction, "path": composition / "rescue_preservation" / f"{direction}_predictions.jsonl"},
        ])
    return entries


def delta(old: dict[str, Any] | None, new: dict[str, Any] | None) -> dict[str, Any] | None:
    if not old or not new:
        return None
    pairs = {"BLEU": "BLEU", "chrF": "chrF++", "ROUGE-L": "ROUGE-L"}
    return {metric: new[new_key] - old[metric] for metric, new_key in pairs.items() if isinstance(old.get(metric), (int, float))}


def protocol_markdown() -> str:
    return """# Unified Translation Quality Metric Protocol

Evaluator source: user-provided `evaluate_gemma_e4b.py`, reproduced in `unified_evaluator.py`.

## BLEU

SacreBLEU `BLEU(tokenize=\"none\", effective_order=True).sentence_score()` is run once per sample, then arithmetic-mean aggregated. Tokenization is selected from the reference language: `jieba.cut(clean(text))` for Chinese; `TIBETAN_SEPARATORS = r\"[\\s་།༎༏༐༑༔]+\"` for Tibetan; whitespace tokens otherwise.

## chrF++

SacreBLEU `CHRF(word_order=2).sentence_score()` is run once per sample, then arithmetic-mean aggregated.

## ROUGE-L

`rouge_score.rouge_scorer.RougeScorer([\"rougeL\"], tokenizer=LangAwareTokenizer())` computes per-sample `fmeasure`, then arithmetic-mean aggregates it. The tokenizer is the same language-aware Chinese/Tibetan tokenizer described for BLEU.

## Retained Metrics and Integrity

`strict_term_pass` and `canonical_term_pass` are read directly from existing prediction JSONLs; no checker is rerun. Every source JSONL has SHA-256 recorded before and after reading, and every available method is required to contain exactly 2,000 unique record IDs per direction.
"""


def write_reports(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    (output / "alignment_checks").mkdir(exist_ok=True)
    (output / "per_method").mkdir(exist_ok=True)
    records: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    alignment: dict[str, list[dict[str, Any]]] = {}
    for spec in registry():
        key = spec["experiment"].replace(" ", "_").lower()
        item = {"experiment": spec["experiment"], "method": spec["method"], "direction": spec["direction"]}
        if "missing" in spec:
            item.update(unavailable(spec["missing"], spec["reason"]))
            item["old_metrics"] = old_metric_source(spec["experiment"], spec["direction"], spec["method"])
        else:
            path = spec["path"]
            try:
                rows, source = load_source(path)
                item.update({"status": "COMPUTED", "source": source, "alignment": validate_rows(rows, expected_count=2000), "metrics": evaluate_rows(rows)})
                item["old_metrics"] = old_metric_source(spec["experiment"], spec["direction"], spec["method"])
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                item.update(unavailable(path, str(error)))
                item["old_metrics"] = old_metric_source(spec["experiment"], spec["direction"], spec["method"])
        item["old_to_unified_delta_raw"] = delta(item.get("old_metrics"), item.get("metrics"))
        records.append(item)
        alignment.setdefault(key, []).append({field: item.get(field) for field in ("experiment", "method", "direction", "status", "source", "alignment", "reason")})
        comparisons.append({field: item.get(field) for field in ("experiment", "method", "direction", "status", "old_metrics", "metrics", "old_to_unified_delta_raw", "reason")})
        method_dir = output / "per_method" / key / spec["method"]
        method_dir.mkdir(parents=True, exist_ok=True)
        (method_dir / f"{spec['direction']}.json").write_text(json.dumps(item, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, items in alignment.items():
        (output / "alignment_checks" / f"{name}_alignment.json").write_text(json.dumps(items, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {"protocol": "evaluate_gemma_e4b.py sentence-level macro averages", "records": records, "computed_record_count": sum(item["status"] == "COMPUTED" for item in records), "unavailable_record_count": sum(item["status"] != "COMPUTED" for item in records)}
    (output / "unified_quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "old_vs_unified_metric_diff.json").write_text(json.dumps({"note": "Raw deltas compare different protocols and are explanatory only; ROUGE-L also changes from old 0..100 scale to unified 0..1 scale.", "records": comparisons}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Unified Translation Quality Report", "", "All computed quality metrics use the direct `evaluate_gemma_e4b.py` protocol. Strict and canonical TermAcc are retained from existing prediction fields. `UNAVAILABLE_SOURCE` means no complete prediction JSONL exists in the current workspace; no score was inferred from historical reports.", "", "| Method | Direction | BLEU | chrF++ | ROUGE-L | Strict TermAcc | Canonical TermAcc | Status |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for item in records:
        metrics = item.get("metrics", {})
        if item["status"] == "COMPUTED":
            lines.append(f"| {item['experiment']} / {item['method']} | {item['direction']} | {metrics['BLEU']:.6f} | {metrics['chrF++']:.6f} | {metrics['ROUGE-L']:.6f} | {metrics['strict_term_accuracy']:.2%} | {metrics['canonical_term_accuracy']:.2%} | COMPUTED |")
        else:
            lines.append(f"| {item['experiment']} / {item['method']} | {item['direction']} | N/A | N/A | N/A | N/A | N/A | UNAVAILABLE_SOURCE |")
    lines += ["", "## Availability", "", f"Computed method-direction records: `{report['computed_record_count']}`. Unavailable method-direction records: `{report['unavailable_record_count']}`.", "", "All available sources require exactly 2,000 unique record IDs for each direction. Detailed count, duplicate, missing, source hash, and source-path checks are in `alignment_checks/`; per-method inputs and scores are in `per_method/`.", "", "## Old vs Unified", "", "`old_vs_unified_metric_diff.json` retains old values and raw differences for methods with both values. These deltas are not performance deltas because the metric definitions differ; they exist for table-audit traceability."]
    (output / "unified_quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "metric_protocol.md").write_text(protocol_markdown(), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    return parser.parse_args()


def main() -> None:
    report = write_reports(parse_args().output_dir)
    print(json.dumps({"computed_record_count": report["computed_record_count"], "unavailable_record_count": report["unavailable_record_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
