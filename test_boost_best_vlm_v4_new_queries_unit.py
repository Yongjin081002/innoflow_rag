import unittest

from test_boost_best_vlm_v4_new_queries import (
    build_table_rows,
    compute_summary,
    load_new_vlm_queries,
)


class BoostBestVlmV4NewQueriesHelpersTest(unittest.TestCase):
    def test_load_new_vlm_queries_returns_20_items(self):
        queries = load_new_vlm_queries()
        self.assertEqual(len(queries), 20)
        self.assertEqual(queries[0]["expected"], "차1-1")
        self.assertEqual(queries[-1]["expected"], "차7-1")

    def test_build_table_rows_marks_top1_top3_and_top5(self):
        details = [
            {"name": "Q1", "expected": "차1-1", "top1": "차1-1", "rank": 1, "score": 0.91},
            {"name": "Q2", "expected": "차2-1", "top1": "차9-9", "rank": 3, "score": 0.72},
            {"name": "Q3", "expected": "차3-1", "top1": "차8-8", "rank": 5, "score": 0.63},
            {"name": "Q4", "expected": "차4-1", "top1": "차7-7", "rank": 999, "score": 0.12},
        ]

        rows = build_table_rows(details)

        self.assertEqual([row["judgment"] for row in rows], ["O", "T3", "T5", "X"])
        self.assertEqual(rows[1]["rank"], 3)
        self.assertEqual(rows[3]["top1"], "차7-7")

    def test_compute_summary_counts_top1_top3_top5(self):
        details = [
            {"rank": 1, "score": 0.91},
            {"rank": 2, "score": 0.71},
            {"rank": 5, "score": 0.51},
            {"rank": 999, "score": 0.11},
        ]

        summary = compute_summary(details)

        self.assertEqual(summary["total"], 4)
        self.assertEqual(summary["top1"], 1)
        self.assertEqual(summary["top3"], 2)
        self.assertEqual(summary["top5"], 3)
        self.assertAlmostEqual(summary["avg_score"], 0.56, places=6)


if __name__ == "__main__":
    unittest.main()
