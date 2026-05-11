import unittest
from collections import Counter

from compare_boost_v3 import GROUP_C
from test_vlm_new_queries import NEW_VLM_QUERIES
from vlm_training_data_v6 import hard_negative_pairs_v6, vlm_pairs_v6


TARGET_MIN_COUNTS = {
    "차41-1": 7,
    "차3-1": 7,
    "차4-1": 7,
    "차31-2": 7,
    "차11-2": 7,
    "차15-1": 7,
}

TARGET_HARD_NEGATIVE_COUNTS = {
    ("차3-1", "차5-2"): 5,
    ("차4-1", "차13-2"): 5,
    ("차31-2", "차7-2"): 5,
    ("차11-2", "차12-1"): 5,
    ("차15-1", "차16-3"): 5,
}


class BoostVlmV6AlignmentTest(unittest.TestCase):
    def test_new_vlm_queries_use_corrected_expected_labels(self):
        query_by_name = {item["name"]: item for item in NEW_VLM_QUERIES}

        self.assertEqual(query_by_name["N13-고속도로추돌(새표현)"]["expected"], "차41-1")
        self.assertEqual(query_by_name["N18-개문사고(새표현)"]["expected"], "차52-1")

    def test_group_c4_query_is_rewritten_to_match_chunk_51_2(self):
        group_c_by_name = {item["name"]: item for item in GROUP_C}
        query = group_c_by_name["C4-아파트주차장통로교차지점충돌"]["query"]

        self.assertEqual(group_c_by_name["C4-아파트주차장통로교차지점충돌"]["expected"], "차51-2")
        self.assertIn("주차", query)
        self.assertIn("추월", query)

    def test_v6_pairs_cover_all_requested_targets(self):
        counts = Counter(item["positive"] for item in vlm_pairs_v6)
        for chunk_id, min_count in TARGET_MIN_COUNTS.items():
            self.assertGreaterEqual(counts[chunk_id], min_count, chunk_id)

    def test_v6_hard_negative_pairs_match_requested_confusions(self):
        counts = Counter(
            (item["positive"], item["negative"]) for item in hard_negative_pairs_v6
        )
        self.assertEqual(counts, TARGET_HARD_NEGATIVE_COUNTS)


if __name__ == "__main__":
    unittest.main()
