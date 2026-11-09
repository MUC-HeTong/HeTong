#!/usr/bin/env python3
"""Preflight checks for the isolated R5 RAG-LoRA Final2000 experiment."""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

from build_prompt import build_prompt_payload, find_forbidden_keys
from prepare_inputs import read_jsonl, runtime_schema_errors, to_runtime_record, validate_alignment

ROOT = Path(__file__).resolve().parents[1]
TERMINOLOGY_ROOT = Path(os.environ.get("BATA_TERMINOLOGY_ROOT", ROOT.parent)).expanduser().resolve()
BASE_MODEL = Path(os.environ.get("BATA_BASE_MODEL", "")).expanduser().resolve()
FINAL_DIR = Path(os.environ.get("BATA_FINAL_DATA_DIR", ROOT.parent.parent / "data/final2000"))
RETRIEVAL_DIR = Path(os.environ.get("BATA_RETRIEVAL_DIR", TERMINOLOGY_ROOT / "rag_integration_v1/ranking_v2/retrieval"))
ADAPTER_ROOT = Path(os.environ.get("BATA_ADAPTER_ROOT", TERMINOLOGY_ROOT / "stage2_lora_training_v1"))
BO2ZH_ADAPTER = ADAPTER_ROOT / "bo2zh/adapter_epoch1"
ZH2BO_ADAPTER = ADAPTER_ROOT / "zh2bo/adapter_epoch1"
ADAPTERS = {"bo2zh": BO2ZH_ADAPTER, "zh2bo": ZH2BO_ADAPTER}


def adapter_for_direction(direction: str) -> Path:
    try:
        return ADAPTERS[direction]
    except KeyError as error:
        raise ValueError(f"unsupported direction: {direction}") from error


def adapter_readiness(direction: str) -> dict[str, Any]:
    adapter = adapter_for_direction(direction)
    required = [adapter / "adapter_config.json", adapter / "adapter_model.safetensors"]
    missing = [str(path) for path in required if not path.is_file()]
    config: dict[str, Any] = {}
    if not missing:
        config = json.loads(required[0].read_text(encoding="utf-8"))
    base_match = config.get("base_model_name_or_path") == str(BASE_MODEL)
    return {
        "base_model": str(BASE_MODEL), "base_model_exists": BASE_MODEL.is_dir(),
        "adapter_path": str(adapter), "adapter_config_base_model": config.get("base_model_name_or_path"),
        "adapter_loaded": not missing and BASE_MODEL.is_dir() and base_match,
        "missing_files": missing, "base_model_match": base_match,
        "load_semantics": "metadata_ready_only; vLLM loading is deferred until inference",
    }


def audit_runtime_records(records: list[dict[str, Any]], sample_size: int = 100) -> dict[str, Any]:
    if len(records) < sample_size:
        raise ValueError("insufficient records for prompt sample")
    sampled = random.Random(20260909).sample(records, sample_size)
    schema_failures = [{"id": row.get("id"), "errors": runtime_schema_errors(row)} for row in sampled if runtime_schema_errors(row)]
    payloads = [build_prompt_payload(row) for row in sampled]
    forbidden = [path for payload in payloads for path in find_forbidden_keys(payload)]
    fallback_payloads = [build_prompt_payload(row) for row in records if row["retrieval_status"] != "HIT"]
    non_hit_term_lines = [
        payload["id"] for payload in fallback_payloads
        if "请优先正确使用术语：" in payload["user"]
    ]
    return {
        "sample_count": len(sampled), "sample_ids": [row["id"] for row in sampled],
        "schema_failures": schema_failures, "forbidden_key_violations": forbidden,
        "all_non_hit_count": len(fallback_payloads),
        "non_hit_term_line_violations": non_hit_term_lines,
    }


def direction_report(direction: str) -> dict[str, Any]:
    final = read_jsonl(FINAL_DIR / f"{direction}_final2000.jsonl")
    retrieval = read_jsonl(RETRIEVAL_DIR / f"{direction}_frozen_R5.jsonl")
    aligned = validate_alignment(final, retrieval, direction)
    records = [to_runtime_record(row) for row in retrieval]
    counts = Counter(row["retrieval_status"] for row in records)
    audit = audit_runtime_records(records)
    adapter = adapter_readiness(direction)
    passed = (
        aligned == 2000 and not audit["schema_failures"] and not audit["forbidden_key_violations"]
        and not audit["non_hit_term_line_violations"] and adapter["adapter_loaded"]
    )
    return {
        "direction": direction, "final2000_count": len(final), "retrieval_count": len(retrieval),
        "matched_count": aligned, "missing_retrieval_count": 0,
        "HIT_count": counts["HIT"], "fallback_count": len(records) - counts["HIT"],
        "status_counts": dict(sorted(counts.items())), "adapter": adapter,
        "prompt_leakage_audit": audit, "passed": passed,
    }


def build_report() -> dict[str, Any]:
    reports = {direction: direction_report(direction) for direction in ("bo2zh", "zh2bo")}
    return {
        "schema_version": "rag_lora_final2000_preflight_v1",
        "translation_inference_started": False,
        "base_model": str(BASE_MODEL),
        "directions": reports,
        "overall_pass": all(report["passed"] for report in reports.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "preflight_report.json")
    args = parser.parse_args()
    report = build_report()
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "overall_pass": report["overall_pass"], "translation_inference_started": False}))


if __name__ == "__main__":
    main()
