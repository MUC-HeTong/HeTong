import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prepare_inputs import to_runtime_record, validate_alignment
from build_prompt import build_prompt_payload, find_forbidden_keys
from verify_alignment import (
    BO2ZH_ADAPTER,
    ZH2BO_ADAPTER,
    adapter_for_direction,
    audit_runtime_records,
)
from run_inference import make_prediction_row, prediction_schema_errors


class RagLoraPipelineTests(unittest.TestCase):
    def setUp(self):
        self.hit = {
            "record_id": "hit-1", "direction": "bo2zh", "source": "source text",
            "retrieval_status": "HIT", "surface": "Tx source", "target_term": "Tx target", "policy": "R5",
        }
        self.non_hit = {
            "record_id": "miss-1", "direction": "bo2zh", "source": "source text",
            "retrieval_status": "NO_HIT", "surface": None, "target_term": None, "policy": "R5",
        }

    def test_hit_dry_run_prompt_includes_only_retrieved_tx(self):
        payload = build_prompt_payload(to_runtime_record(self.hit))
        self.assertIn("Tx source -> Tx target", payload["user"])
        self.assertEqual(payload["terminology_block"], {
            "retrieved_source_term": "Tx source", "retrieved_target_term": "Tx target",
        })
        self.assertEqual(find_forbidden_keys(payload), [])

    def test_non_hit_prompt_has_no_terminology_line(self):
        payload = build_prompt_payload(to_runtime_record(self.non_hit))
        self.assertNotIn("请优先正确使用术语：", payload["user"])
        self.assertIsNone(payload["terminology_block"])
        self.assertEqual(payload["route"], "NO_TERMINOLOGY_BASE_FALLBACK")

    def test_direction_uses_only_matching_stage2_adapter(self):
        self.assertEqual(adapter_for_direction("bo2zh"), BO2ZH_ADAPTER)
        self.assertEqual(adapter_for_direction("zh2bo"), ZH2BO_ADAPTER)
        self.assertNotEqual(adapter_for_direction("bo2zh"), ZH2BO_ADAPTER)

    def test_runtime_schema_and_sample_audit_are_gold_isolated(self):
        records = [to_runtime_record(self.hit), to_runtime_record(self.non_hit)] * 50
        audit = audit_runtime_records(records, sample_size=100)
        self.assertEqual(audit["sample_count"], 100)
        self.assertEqual(audit["schema_failures"], [])
        self.assertEqual(audit["forbidden_key_violations"], [])
        self.assertEqual(audit["non_hit_term_line_violations"], [])

    def test_one_hundred_dry_run_prediction_rows_follow_required_schema(self):
        records = [to_runtime_record(self.hit), to_runtime_record(self.non_hit)] * 50
        rows = [make_prediction_row(record, "prediction", "reference") for record in records]
        self.assertEqual(len(rows), 100)
        self.assertEqual([prediction_schema_errors(row) for row in rows], [[]] * 100)

    def test_alignment_requires_ordered_id_direction_and_source_equality(self):
        final = [{"record_id": str(index), "direction": "bo2zh", "source": f"x-{index}"} for index in range(2000)]
        retrieval = [row.copy() for row in final]
        self.assertEqual(validate_alignment(final, retrieval, "bo2zh"), 2000)
        with self.assertRaisesRegex(ValueError, "alignment"):
            changed = [row.copy() for row in final]
            changed[0]["source"] = "changed"
            validate_alignment(final, changed, "bo2zh")


if __name__ == "__main__":
    unittest.main()
