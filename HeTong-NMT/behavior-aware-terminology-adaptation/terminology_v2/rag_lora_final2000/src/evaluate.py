#!/usr/bin/env python3
"""Post-generation RAG-LoRA evaluation using the retained unified metrics."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TERMINOLOGY_ROOT = Path(os.environ.get("BATA_TERMINOLOGY_ROOT", ROOT.parent)).expanduser().resolve()
FINAL_DIR = Path(os.environ.get("BATA_FINAL_DATA_DIR", ROOT.parent.parent / "data/final2000"))
UNIFIED = TERMINOLOGY_ROOT / "unified_quality_evaluation_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical_forms(row: dict[str, Any]) -> list[str]:
    if row["direction"] == "bo2zh":
        return [row["zh_term"]]
    canonical = row["bo_term"].strip().rstrip("།")
    return [canonical, row["bo_term"]]


def evaluation_rows(direction: str, predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final = read_jsonl(FINAL_DIR / f"{direction}_final2000.jsonl")
    expected_ids = [str(row["record_id"]) for row in final]
    actual_ids = [str(row["id"]) for row in predictions]
    if actual_ids != expected_ids:
        raise ValueError("prediction IDs do not match frozen Final2000 order")
    rows = []
    for prediction, gold in zip(predictions, final):
        text = prediction["prediction"]
        strict = gold["target_term"] in text
        canonical = any(form and form in text for form in canonical_forms(gold))
        rows.append({"record_id": gold["record_id"], "direction": direction, "reference": gold["reference"],
                     "prediction": text, "strict_term_pass": strict, "canonical_term_pass": canonical})
    return rows


def calculate_metrics(direction: str, predictions: list[dict[str, Any]]) -> dict[str, Any]:
    sys.path.insert(0, str(UNIFIED))
    from unified_evaluator import evaluate_rows
    rows = evaluation_rows(direction, predictions)
    scores = evaluate_rows(rows)
    hit_count = sum(row["retrieval_status"] == "HIT" for row in predictions)
    return {
        "sample_count": scores["sample_count"], "HIT_count": hit_count,
        "fallback_count": len(predictions) - hit_count,
        "Strict TermAcc": scores["strict_term_accuracy"] * 100,
        "Canonical TermAcc": scores["canonical_term_accuracy"] * 100,
        "BLEU": scores["BLEU"], "chrF++": scores["chrF++"], "ROUGE-L": scores["ROUGE-L"] * 100,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate completed RAG-LoRA predictions after Final2000 inference.")
    parser.add_argument("--direction", required=True, choices=("bo2zh", "zh2bo"))
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    predictions = args.predictions or ROOT / args.direction / "predictions.jsonl"
    if not predictions.is_file():
        raise FileNotFoundError(f"predictions are unavailable; inference has not completed: {predictions}")
    metrics = calculate_metrics(args.direction, read_jsonl(predictions))
    output = args.output or ROOT / args.direction / "metrics.json"
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"direction": args.direction, "metrics": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
