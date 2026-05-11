import unittest
from collections import Counter

from benchmark_boost_vlm_v5 import build_chronic_item_report
from vlm_training_data_v5 import hard_negative_pairs_v5, vlm_pairs_v5


TARGET_MIN_COUNTS = {
    "차11-2": 7,
    "차44-1": 7,
    "차61-1": 7,
    "차3-1": 7,
    "차4-1": 7,
    "차31-2": 7,
    "차33-2": 7,
    "차42-1": 7,
    "차51-2": 7,
}

TARGET_HARD_NEGATIVE_COUNTS = {
    ("차11-2", "차12-1"): 5,
    ("차44-1", "차41-1"): 5,
    ("차61-1", "차42-3"): 5,
    ("차33-2", "차33-1"): 5,
    ("차42-1", "차41-1"): 5,
    ("차51-2", "차51-1"): 5,
}


class BoostVlmV5DataTest(unittest.TestCase):
    def test_v5_pairs_cover_all_requested_weak_targets(self):
        counts = Counter(item["positive"] for item in vlm_pairs_v5)

        for chunk_id, min_count in TARGET_MIN_COUNTS.items():
            self.assertGreaterEqual(counts[chunk_id], min_count, chunk_id)

    def test_v5_hard_negative_pairs_match_requested_confusions(self):
        counts = Counter(
            (item["positive"], item["negative"]) for item in hard_negative_pairs_v5
        )

        self.assertEqual(
            counts,
            TARGET_HARD_NEGATIVE_COUNTS,
        )

    def test_build_chronic_item_report_marks_fix_improve_and_same(self):
        baseline = {
            "Q1": {"expected": "차11-2", "top1": "차12-1", "rank": 999, "score": -0.1673},
            "Q2": {"expected": "차42-1", "top1": "차41-1", "rank": 3, "score": 0.6312},
            "Q3": {"expected": "차33-2", "top1": "차33-1", "rank": 2, "score": 0.5871},
        }
        current = {
            "Q1": {"expected": "차11-2", "top1": "차11-2", "rank": 1, "score": 0.4123},
            "Q2": {"expected": "차42-1", "top1": "차42-1", "rank": 1, "score": 0.7444},
            "Q3": {"expected": "차33-2", "top1": "차33-1", "rank": 2, "score": 0.6021},
        }

        rows = build_chronic_item_report(
            target_items=[
                ("차11-2", "Q1"),
                ("차42-1", "Q2"),
                ("차33-2", "Q3"),
            ],
            baseline_details=baseline,
            current_details=current,
        )

        self.assertEqual(rows[0]["status"], "FIXED")
        self.assertEqual(rows[1]["status"], "FIXED")
        self.assertEqual(rows[2]["status"], "UNCHANGED")


if __name__ == "__main__":
    unittest.main()
