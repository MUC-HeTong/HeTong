"""Tx-only prompt rendering for the RAG-LoRA runtime."""
from __future__ import annotations

from typing import Any

SYSTEM = {
    "bo2zh": "你是专业的藏汉翻译助手。请准确、完整地将藏文翻译成中文，只输出中文译文，不要解释。",
    "zh2bo": "你是专业的汉藏翻译助手。请准确、完整地将中文翻译成藏文，只输出藏文译文，不要解释。",
}
INSTRUCTION = {
    "bo2zh": "请将以下藏文翻译成中文，只输出译文。",
    "zh2bo": "请将以下中文翻译成藏文，只输出译文。",
}
FORBIDDEN_KEYS = frozenset({
    "source_term", "target_term", "bo_term", "zh_term", "reference", "translation",
    "gold", "canonical_target_term", "verified_terms",
})


def build_prompt_payload(record: dict[str, Any]) -> dict[str, Any]:
    hit = record["retrieval_status"] == "HIT"
    user = INSTRUCTION[record["direction"]] + "\n"
    block = None
    if hit:
        block = {
            "retrieved_source_term": record["retrieved_source_term"],
            "retrieved_target_term": record["retrieved_target_term"],
        }
        user += f"请优先正确使用术语：{block['retrieved_source_term']} -> {block['retrieved_target_term']}\n"
    user += f"原文：{record['source']}"
    return {
        "id": record["id"],
        "direction": record["direction"],
        "retrieval_status": record["retrieval_status"],
        "system": SYSTEM[record["direction"]],
        "user": user,
        "terminology_block": block,
        "route": "RAG_LORA_HIT" if hit else "NO_TERMINOLOGY_BASE_FALLBACK",
    }


def find_forbidden_keys(value: Any, path: str = "$") -> list[str]:
    problems: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in FORBIDDEN_KEYS or key.startswith("gold_"):
                problems.append(child_path)
            problems.extend(find_forbidden_keys(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            problems.extend(find_forbidden_keys(child, f"{path}[{index}]"))
    return problems
