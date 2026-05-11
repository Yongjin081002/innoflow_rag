import unittest
from collections import Counter

from compare_boost_v3 import GROUP_B
from test_vlm_new_queries import NEW_VLM_QUERIES
from vlm_training_data_v7 import hard_negative_pairs_v7, vlm_pairs_v7


TARGET_MIN_COUNTS = {
    "차3-1": 10,
    "차4-1": 10,
    "차31-2": 10,
    "차11-2": 10,
    "차1-1": 10,
    "차15-1": 10,
}

TARGET_HARD_NEGATIVE_COUNTS = {
    ("차3-1", "차5-2"): 7,
    ("차4-1", "차13-2"): 7,
    ("차31-2", "차7-2"): 7,
    ("차11-2", "차12-1"): 7,
    ("차1-1", "차31-1"): 7,
    ("차15-1", "차16-3"): 7,
}


class BoostVlmV7AlignmentTest(unittest.TestCase):
    def test_new_vlm_queries_keep_corrected_expected_labels(self):
        query_by_name = {item["name"]: item for item in NEW_VLM_QUERIES}

        self.assertEqual(query_by_name["N13-고속도로추돌(새표현)"]["expected"], "차41-1")
        self.assertEqual(query_by_name["N18-개문사고(새표현)"]["expected"], "차52-1")

    def test_group_b_queries_align_with_chunk_semantics(self):
        group_b_by_name = {item["name"]: item for item in GROUP_B}

        b2 = group_b_by_name["B2-적색우회전녹색직진교차충돌"]
        self.assertEqual(b2["expected"], "차3-1")
        self.assertIn("좌회전 화살표", b2["query"])

        b3 = group_b_by_name["B3-동일방향직진우회전교차충돌"]
        self.assertEqual(b3["expected"], "차4-1")
        self.assertIn("맞은편에서 우회전", b3["query"])
        self.assertTrue("녹색 화살표" in b3["query"] or "좌회전 신호" in b3["query"])

        b5 = group_b_by_name["B5-일방통행로역주행정면충돌"]
        self.assertEqual(b5["expected"], "차31-2")
        self.assertIn("도로가 아닌 장소", b5["query"])
        self.assertIn("중앙선", b5["query"])

        b9 = group_b_by_name["B9-비신호동일폭교차로우측차우선위반"]
        self.assertEqual(b9["expected"], "차11-2")
        self.assertIn("노면표시", b9["query"])
        self.assertIn("직진 전용 차로", b9["query"])

    def test_v7_pairs_cover_all_requested_targets(self):
        counts = Counter(item["positive"] for item in vlm_pairs_v7)
        for chunk_id, min_count in TARGET_MIN_COUNTS.items():
            self.assertGreaterEqual(counts[chunk_id], min_count, chunk_id)

    def test_v7_pairs_embed_required_phrases(self):
        by_chunk = {}
        for item in vlm_pairs_v7:
            by_chunk.setdefault(item["positive"], []).append(item["query"])

        for query in by_chunk["차3-1"]:
            self.assertIn("좌회전 화살표", query)

        for query in by_chunk["차4-1"]:
            self.assertIn("맞은편에서 우회전", query)
            self.assertTrue("녹색 화살표" in query or "좌회전 신호" in query)

        for query in by_chunk["차31-2"]:
            self.assertIn("도로가 아닌 장소", query)
            self.assertIn("중앙선", query)

        for query in by_chunk["차11-2"]:
            self.assertIn("노면표시", query)
            self.assertIn("직진 전용 차로", query)

        for query in by_chunk["차1-1"]:
            self.assertIn("신호등", query)
            self.assertIn("녹색신호", query)
            self.assertIn("적색신호", query)

        for query in by_chunk["차15-1"]:
            self.assertIn("신호 없는 교차로", query)

    def test_v7_hard_negative_pairs_match_requested_confusions(self):
        counts = Counter(
            (item["positive"], item["negative"]) for item in hard_negative_pairs_v7
        )
        self.assertEqual(counts, TARGET_HARD_NEGATIVE_COUNTS)


if __name__ == "__main__":
    unittest.main()
