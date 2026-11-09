"""Build source-only runtime records from frozen R5 retrieval output."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

NORMALIZED_STATUS = {
    "HIT": "HIT",
    "NO_HIT": "NO_HIT",
    "AMBIGUOUS_MATCH": "AMBIGUOUS_MATCH",
    "BIDIRECTIONAL_CONFLICT_BLOCKED": "BLOCKED",
}
RUNTIME_FIELDS = (
    "id", "direction", "source", "retrieval_status",
    "retrieved_source_term", "retrieved_target_term",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def to_runtime_record(frozen: dict[str, Any]) -> dict[str, Any]:
    status = NORMALIZED_STATUS[frozen["retrieval_status"]]
    hit = status == "HIT"
    source_term = frozen.get("surface") if hit else None
    target_term = frozen.get("target_term") if hit else None
    if hit and (not isinstance(source_term, str) or not source_term or not isinstance(target_term, str) or not target_term):
        raise ValueError(f"HIT {frozen['record_id']} lacks retrieved Tx")
    return {
        "id": str(frozen["record_id"]),
        "direction": frozen["direction"],
        "source": frozen["source"],
        "retrieval_status": status,
        "retrieved_source_term": source_term,
        "retrieved_target_term": target_term,
    }


def validate_alignment(final_rows: list[dict[str, Any]], retrieval_rows: list[dict[str, Any]], direction: str) -> int:
    expected = [(str(row["record_id"]), row["direction"], row["source"]) for row in final_rows]
    actual = [(str(row["record_id"]), row["direction"], row["source"]) for row in retrieval_rows]
    if len(expected) != 2000 or len(actual) != 2000:
        raise ValueError(f"{direction}: expected exactly 2000 Final2000 and R5 rows")
    if any(row[1] != direction for row in expected + actual):
        raise ValueError(f"{direction}: direction field mismatch")
    if actual != expected:
        raise ValueError(f"{direction}: R5 and Final2000 order/source alignment failed")
    return len(expected)


def runtime_schema_errors(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(record) != set(RUNTIME_FIELDS):
        errors.append("runtime fields differ from source-only schema")
    if record.get("direction") not in {"bo2zh", "zh2bo"}:
        errors.append("invalid direction")
    if not isinstance(record.get("id"), str) or not record.get("id"):
        errors.append("invalid id")
    if not isinstance(record.get("source"), str) or not record.get("source"):
        errors.append("invalid source")
    if record.get("retrieval_status") not in {"HIT", "NO_HIT", "AMBIGUOUS_MATCH", "BLOCKED"}:
        errors.append("invalid retrieval_status")
    terms = (record.get("retrieved_source_term"), record.get("retrieved_target_term"))
    if record.get("retrieval_status") == "HIT":
        if not all(isinstance(term, str) and term for term in terms):
            errors.append("HIT has missing Tx")
    elif any(term is not None for term in terms):
        errors.append("non-HIT has injected terminology")
    return errors
