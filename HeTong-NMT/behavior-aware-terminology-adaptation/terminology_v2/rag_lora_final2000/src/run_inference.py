#!/usr/bin/env python3
"""Deferred vLLM runner for R5-conditioned Stage2 LoRA generation.

This module is intentionally not invoked by the preparation stage. It refuses
to run unless the persisted preflight report is fully passing.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

from build_prompt import build_prompt_payload
from prepare_inputs import read_jsonl, to_runtime_record, validate_alignment
from verify_alignment import BASE_MODEL, FINAL_DIR, RETRIEVAL_DIR, adapter_for_direction

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "preflight_report.json"
PREDICTION_FIELDS = (
    "id", "direction", "source", "retrieved_source_term", "retrieved_target_term",
    "retrieval_status", "prediction", "reference",
)


def require_preflight() -> None:
    report = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if not report.get("overall_pass") or report.get("translation_inference_started"):
        raise RuntimeError("preflight did not pass or an inference run is already recorded; refusing inference")


def load_records(direction: str) -> list[dict[str, Any]]:
    rows = read_jsonl(RETRIEVAL_DIR / f"{direction}_frozen_R5.jsonl")
    return [to_runtime_record(row) for row in rows]


def build_tokenizer_shim() -> Path:
    """Make checkpoint tokenizer metadata readable by the retained vLLM environment."""
    target = ROOT / "tokenizer_compat"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "chat_template.jinja"):
        shutil.copy2(BASE_MODEL / name, target / name)
    config = json.loads((BASE_MODEL / "tokenizer_config.json").read_text(encoding="utf-8"))
    config["tokenizer_class"] = "PreTrainedTokenizerFast"
    config.pop("backend", None)
    config.pop("extra_special_tokens", None)
    (target / "tokenizer_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def render_chat(tokenizer: Any, record: dict[str, Any]) -> str:
    payload = build_prompt_payload(record)
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": payload["system"]}, {"role": "user", "content": payload["user"]}],
        tokenize=False, add_generation_prompt=True,
    )


def make_prediction_row(record: dict[str, Any], text: str, reference: str) -> dict[str, Any]:
    return {
        "id": record["id"], "direction": record["direction"], "source": record["source"],
        "retrieved_source_term": record["retrieved_source_term"], "retrieved_target_term": record["retrieved_target_term"],
        "retrieval_status": record["retrieval_status"], "prediction": text, "reference": reference,
    }


def prediction_schema_errors(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(row) != set(PREDICTION_FIELDS):
        errors.append("prediction fields differ from required schema")
    if not all(isinstance(row.get(key), str) and row[key] for key in ("id", "direction", "source", "prediction", "reference")):
        errors.append("required prediction text field is missing")
    if row.get("retrieval_status") == "HIT":
        if not all(isinstance(row.get(key), str) and row[key] for key in ("retrieved_source_term", "retrieved_target_term")):
            errors.append("HIT prediction lacks Tx")
    elif row.get("retrieved_source_term") is not None or row.get("retrieved_target_term") is not None:
        errors.append("non-HIT prediction contains Tx")
    return errors


def run_direction(direction: str, batch_size: int, max_samples: int | None, output: Path | None, run_report: Path | None) -> None:
    require_preflight()
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive")
    # Delayed imports guarantee preflight is the first gate before GPU/model use.
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    records = load_records(direction)
    final_rows = read_jsonl(FINAL_DIR / f"{direction}_final2000.jsonl")
    retrieval_rows = read_jsonl(RETRIEVAL_DIR / f"{direction}_frozen_R5.jsonl")
    validate_alignment(final_rows, retrieval_rows, direction)
    if max_samples is not None:
        records = records[:max_samples]
    references = {str(row["record_id"]): row["reference"] for row in final_rows}
    adapter = adapter_for_direction(direction)
    tokenizer_shim = build_tokenizer_shim()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_shim, trust_remote_code=True)
    llm = LLM(model=str(BASE_MODEL), tokenizer=str(tokenizer_shim), dtype="float16", trust_remote_code=True, max_model_len=4096, enable_lora=True)
    params = SamplingParams(temperature=0.0, top_p=1.0, repetition_penalty=1.0, max_tokens=1024)
    request = LoRARequest(f"rag_lora_{direction}", 1, str(adapter))
    output = output or ROOT / direction / "predictions.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size:
        raise RuntimeError(f"refusing to overwrite existing predictions: {output}")

    started = time.monotonic()
    prediction_lengths: list[int] = []
    empty_output_count = 0
    with output.open("w", encoding="utf-8") as handle:
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            prompts = [render_chat(tokenizer, row) for row in batch]
            generated_by_position: dict[int, str] = {}
            # HIT gets its matching direction adapter; non-HIT is strictly base-only.
            for route, lora_request in ((True, request), (False, None)):
                positions = [index for index, row in enumerate(batch) if (row["retrieval_status"] == "HIT") == route]
                if not positions:
                    continue
                outputs = llm.generate([prompts[index] for index in positions], params, lora_request=lora_request, use_tqdm=False)
                for index, generated in zip(positions, outputs):
                    generated_by_position[index] = generated.outputs[0].text.strip()
            for index, row in enumerate(batch):
                prediction = make_prediction_row(row, generated_by_position[index], references[row["id"]])
                if errors := prediction_schema_errors(prediction):
                    raise RuntimeError(f"invalid prediction schema for {row['id']}: {errors}")
                prediction_lengths.append(len(prediction["prediction"]))
                empty_output_count += not bool(prediction["prediction"])
                handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    elapsed = time.monotonic() - started
    report = {
        "direction": direction, "sample_count": len(records), "generation_success_count": len(records) - empty_output_count,
        "empty_output_count": empty_output_count,
        "average_prediction_character_length": sum(prediction_lengths) / len(prediction_lengths),
        "elapsed_seconds": elapsed, "samples_per_second": len(records) / elapsed if elapsed else None,
        "base_model": str(BASE_MODEL), "adapter_path": str(adapter), "adapter_loaded": True,
        "output": str(output),
    }
    if run_report:
        run_report.parent.mkdir(parents=True, exist_ok=True)
        run_report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", required=True, choices=("bo2zh", "zh2bo"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-report", type=Path)
    args = parser.parse_args()
    run_direction(args.direction, args.batch_size, args.max_samples, args.output, args.run_report)


if __name__ == "__main__":
    main()
