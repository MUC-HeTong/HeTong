import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unified_evaluator import (  # noqa: E402
    normalize_condition_rows,
    evaluate_rows,
    tibetan_tokens,
    validate_id_alignment,
    validate_rows,
)


class UnifiedEvaluatorTest(unittest.TestCase):
    def test_tibetan_tokenizer_splits_tsek_and_punctuation(self):
        self.assertEqual(tibetan_tokens("ཀ་ ཁ།ག༎"), ["ཀ", "ཁ", "ག"])

    def test_identical_rows_receive_direct_protocol_scores(self):
        rows = [
            {
                "record_id": "zh-1",
                "direction": "bo2zh",
                "prediction": "和平抗暴",
                "reference": "和平抗暴",
                "strict_term_pass": True,
                "canonical_term_pass": True,
            },
            {
                "record_id": "bo-1",
                "direction": "zh2bo",
                "prediction": "ཞི་བ་ ཡིན།",
                "reference": "ཞི་བ་ ཡིན།",
                "strict_term_pass": False,
                "canonical_term_pass": True,
            },
        ]
        scores = evaluate_rows(rows)
        self.assertEqual(scores["sample_count"], 2)
        self.assertAlmostEqual(scores["BLEU"], 100.0)
        self.assertAlmostEqual(scores["chrF++"], 100.0)
        self.assertAlmostEqual(scores["ROUGE-L"], 1.0)
        self.assertEqual(scores["strict_term_success"], 1)
        self.assertEqual(scores["canonical_term_success"], 2)

    def test_validate_rows_rejects_duplicate_record_ids(self):
        rows = [
            {"record_id": "duplicate", "direction": "bo2zh"},
            {"record_id": "duplicate", "direction": "bo2zh"},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_rows(rows, expected_count=2)

    def test_condition_adapter_reuses_existing_checker_fields(self):
        source = [
            {
                "record_id": "one",
                "direction": "bo2zh",
                "reference": "中文参考",
                "direction_lora_prediction": "中文预测",
                "direction_lora_term_pass": True,
                "direction_lora_canonical_term_pass": False,
            }
        ]
        rows = normalize_condition_rows(source, "direction_lora")
        self.assertEqual(rows[0]["prediction"], "中文预测")
        self.assertTrue(rows[0]["strict_term_pass"])
        self.assertFalse(rows[0]["canonical_term_pass"])

    def test_alignment_rejects_different_record_id_order(self):
        left = [{"record_id": "a"}, {"record_id": "b"}]
        right = [{"record_id": "b"}, {"record_id": "a"}]
        with self.assertRaisesRegex(ValueError, "record_id order"):
            validate_id_alignment(left, right)


if __name__ == "__main__":
    unittest.main()
