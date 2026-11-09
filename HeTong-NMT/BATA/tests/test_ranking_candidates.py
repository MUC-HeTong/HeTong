import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "ranking_v2" / "src"))

from current_kb_exact_retriever import build_index
from ranking_candidates import enumerate_candidates, select_candidate


def entry(term_id, zh, bo, surfaces=(), status="NO_CONFLICT", count=1):
    return {"term_id": term_id, "zh_term": zh, "bo_canonical": bo,
            "bo_surface_forms": list(surfaces), "conflict_status": status,
            "needs_review": status != "NO_CONFLICT", "source_count": count,
            "source_pair_count": count, "source_files": ["test"]}


class RankingCandidateTests(unittest.TestCase):
    def test_enumerate_candidates_rejects_gold_fields_and_keeps_overlaps(self):
        index = build_index([entry("A", "甲", "ཀ"), entry("B", "乙", "ཀ་ཁ")])
        with self.assertRaises(ValueError):
            enumerate_candidates({"record_id": "1", "direction": "bo2zh", "source": "ཀ་ཁ", "target_term": "甲"}, index)
        spans = enumerate_candidates({"record_id": "1", "direction": "bo2zh", "source": "ཀ་ཁ"}, index)
        self.assertEqual([span["surface"] for span in spans], ["ཀ", "ཀ་ཁ"])

    def test_r2_prefers_shortest_eligible_span(self):
        index = build_index([entry("A", "甲", "ཀ"), entry("B", "乙", "ཀ་ཁ")])
        spans = enumerate_candidates({"record_id": "1", "direction": "bo2zh", "source": "ཀ་ཁ"}, index)
        self.assertEqual(select_candidate(spans, "R2")["surface"], "ཀ")

    def test_r3_prefers_canonical_before_alias(self):
        index = build_index([entry("A", "甲", "ཀ", ["ཀ་ཁ"]), entry("B", "乙", "ཀ་ཁ")])
        spans = enumerate_candidates({"record_id": "1", "direction": "bo2zh", "source": "ཀ་ཁ"}, index)
        self.assertEqual(select_candidate(spans, "R3")["match_type"], "canonical")

    def test_r5_prefers_complete_chinese_boundaries(self):
        index = build_index([entry("A", "甲", "ཀ"), entry("B", "术语", "乙")])
        spans = enumerate_candidates({"record_id": "1", "direction": "zh2bo", "source": "汉术语。术语"}, index)
        self.assertEqual(select_candidate(spans, "R5")["start"], 4)

    def test_r6_is_a_source_only_named_hybrid(self):
        index = build_index([entry("A", "甲", "ཀ", ["ཀ་ཁ"], count=1), entry("B", "乙", "ཀ་ཁ", count=9)])
        spans = enumerate_candidates({"record_id": "1", "direction": "bo2zh", "source": "ཀ་ཁ"}, index)
        self.assertEqual(select_candidate(spans, "R6")["match_type"], "canonical")
