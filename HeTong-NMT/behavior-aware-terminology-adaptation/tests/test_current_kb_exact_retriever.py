import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from current_kb_exact_retriever import audit_index, build_index, retrieve, to_retriever_input


def entry(term_id, zh_term, bo_canonical, aliases=(), conflict_status="NO_CONFLICT", source_count=1, source_pair_count=1):
    return {
        "term_id": term_id,
        "zh_term": zh_term,
        "bo_canonical": bo_canonical,
        "bo_surface_forms": list(aliases),
        "conflict_status": conflict_status,
        "needs_review": conflict_status != "NO_CONFLICT",
        "source_count": source_count,
        "source_pair_count": source_pair_count,
        "source_files": ["fixture.json"],
    }


class CurrentKbExactRetrieverTests(unittest.TestCase):
    def test_bo2zh_canonical_returns_standardized_chinese_target(self):
        index = build_index([entry("TERM_1", "规范中文", "བོ་ཀ")])
        row = retrieve({"record_id": "1", "direction": "bo2zh", "source": "བོ་ཀ 示例"}, index)
        self.assertEqual(row["retrieval_status"], "HIT")
        self.assertEqual(row["target_term"], "规范中文")
        self.assertEqual(row["match_type"], "canonical")

    def test_bo2zh_alias_returns_standardized_chinese_target(self):
        index = build_index([entry("TERM_1", "规范中文", "བོ་ཀ", ["བོ་ཀ།"])])
        row = retrieve({"record_id": "2", "direction": "bo2zh", "source": "བོ་ཀ། 示例"}, index)
        self.assertEqual(row["retrieval_status"], "HIT")
        self.assertEqual(row["target_term"], "规范中文")
        self.assertEqual(row["match_type"], "alias")

    def test_canonical_only_index_excludes_observed_alias_surfaces(self):
        index = build_index([entry("TERM_1", "规范中文", "བོ་ཀ", ["བོ་ཀ།"])], include_observed_surfaces=False)
        row = retrieve({"record_id": "2a", "direction": "bo2zh", "source": "བོ་ཀ། 示例"}, index)
        self.assertEqual(row["matched_source_surface"], "བོ་ཀ")
        self.assertEqual(row["match_type"], "canonical")

    def test_zh2bo_returns_canonical_tibetan_and_all_surface_targets(self):
        index = build_index([entry("TERM_1", "规范中文", "བོ་ཀ", ["བོ་ཀ།"])])
        row = retrieve({"record_id": "3", "direction": "zh2bo", "source": "规范中文 示例"}, index)
        self.assertEqual(row["retrieval_status"], "HIT")
        self.assertEqual(row["target_term"], "བོ་ཀ")
        self.assertEqual(row["canonical_target_term"], "བོ་ཀ")
        self.assertEqual(row["surface_target_terms"], ["བོ་ཀ", "བོ་ཀ།"])

    def test_shared_alias_with_two_targets_is_ambiguous_and_not_injected(self):
        index = build_index([
            entry("TERM_1", "甲", "ཀ", ["ཀ།"]),
            entry("TERM_2", "乙", "ཁ", ["ཀ།"]),
        ])
        row = retrieve({"record_id": "4", "direction": "bo2zh", "source": "ཀ།"}, index)
        self.assertEqual(row["retrieval_status"], "AMBIGUOUS_MATCH")
        self.assertIsNone(row["target_term"])
        self.assertEqual(row["distinct_target_count"], 2)

    def test_bidirectional_conflict_is_blocked_even_when_surface_has_one_target(self):
        index = build_index([entry("TERM_3", "甲", "ག", conflict_status="BIDIRECTIONAL_CONFLICT")])
        row = retrieve({"record_id": "5", "direction": "bo2zh", "source": "ག"}, index)
        self.assertEqual(row["retrieval_status"], "BIDIRECTIONAL_CONFLICT_BLOCKED")
        self.assertIsNone(row["target_term"])
        self.assertTrue(row["would_be_unique_without_bidirectional_block"])

    def test_opposite_direction_conflict_can_return_unique_target_with_provenance(self):
        index = build_index([entry("TERM_4", "甲", "ང", conflict_status="ZH_TO_BO_CONFLICT")])
        row = retrieve({"record_id": "6", "direction": "bo2zh", "source": "ང"}, index)
        self.assertEqual(row["retrieval_status"], "HIT")
        self.assertEqual(row["conflict_statuses"], ["ZH_TO_BO_CONFLICT"])

    def test_longer_surface_wins_deterministically(self):
        index = build_index([
            entry("TERM_1", "短", "ཀ"),
            entry("TERM_2", "长", "ཀ་ཁ"),
        ])
        row = retrieve({"record_id": "7", "direction": "bo2zh", "source": "ཀ་ཁ"}, index)
        self.assertEqual(row["retrieval_status"], "HIT")
        self.assertEqual(row["target_term"], "长")

    def test_source_only_adapter_strips_gold_metadata(self):
        full = {
            "record_id": "8", "direction": "bo2zh", "source": "x", "reference": "r",
            "source_term": "gold source", "target_term": "gold target",
            "canonical_target_term": "gold target", "surface_target_terms": ["gold target"],
            "terminology_id": "TERM_GOLD",
        }
        self.assertEqual(to_retriever_input(full), {"record_id": "8", "direction": "bo2zh", "source": "x"})

    def test_retrieve_rejects_records_with_gold_fields(self):
        index = build_index([entry("TERM_1", "甲", "ཀ")])
        with self.assertRaises(ValueError):
            retrieve({"record_id": "9", "direction": "bo2zh", "source": "ཀ", "target_term": "甲"}, index)

    def test_index_audit_counts_alias_ambiguity_separately(self):
        index = build_index([
            entry("TERM_1", "甲", "ཀ", ["ཀ།"]),
            entry("TERM_2", "乙", "ཁ", ["ཀ།"]),
        ])
        audit = audit_index(index)
        self.assertEqual(audit["bo2zh"]["alias_source_ambiguities"], 1)
        self.assertEqual(audit["bo2zh"]["source_surfaces_with_multiple_targets"], 1)


if __name__ == "__main__":
    unittest.main()
