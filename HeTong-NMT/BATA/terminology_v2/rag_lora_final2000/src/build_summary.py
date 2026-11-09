#!/usr/bin/env python3
"""Combine completed directional RAG-LoRA metrics into the paper summary schema."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    summary = {}
    for direction in ("bo2zh", "zh2bo"):
        metrics = json.loads((ROOT / direction / "metrics.json").read_text(encoding="utf-8"))
        summary[direction] = {
            "sample_count": metrics["sample_count"], "HIT": metrics["HIT_count"],
            "fallback": metrics["fallback_count"], "Strict_TermAcc": metrics["Strict TermAcc"],
            "Canonical_TermAcc": metrics["Canonical TermAcc"], "BLEU": metrics["BLEU"],
            "chrF++": metrics["chrF++"], "ROUGE-L": metrics["ROUGE-L"],
        }
    (ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
