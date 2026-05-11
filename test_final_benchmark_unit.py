import unittest

from benchmark_final_v4 import (
    aggregate_category_stats,
    build_wrong_answer_rows,
    compute_overfit_gap,
    summarize_details,
)


class BenchmarkHelpersTest(unittest.TestCase):
    def test_summarize_details_counts_topk_and_average(self):
        details = [
            {"rank": 1, "score": 0.9},
            {"rank": 2, "score": 0.6},
            {"rank": 5, "score": 0.4},
            {"rank": 999, "score": 0.1},
        ]

        summary = summarize_details(details)

        self.assertEqual(
            summary,
            {
                "total": 4,
                "top1": 1,
                "top3": 2,
                "top5": 3,
                "avg_score": 0.5,
            },
        )

    def test_aggregate_category_stats_groups_vlm_30_by_category(self):
        details = [
            {"category": "신호교차로", "rank": 1, "score": 0.9},
            {"category": "신호교차로", "rank": 4, "score": 0.3},
            {"category": "추돌", "rank": 2, "score": 0.8},
        ]

        stats = aggregate_category_stats(details)

        self.assertEqual(
            stats["신호교차로"],
            {
                "total": 2,
                "top1": 1,
                "top3": 1,
                "top5": 2,
                "avg_score": 0.6,
            },
        )
        self.assertEqual(
            stats["추돌"],
            {
                "total": 1,
                "top1": 0,
                "top3": 1,
                "top5": 1,
                "avg_score": 0.8,
            },
        )

    def test_compute_overfit_gap_uses_group_a_vs_c_average_score(self):
        details = [
            {"group": "A", "score": 0.82},
            {"group": "A", "score": 0.78},
            {"group": "C", "score": 0.61},
            {"group": "C", "score": 0.59},
        ]

        gap = compute_overfit_gap(details)

        self.assertEqual(
            gap,
            {
                "group_a_avg_score": 0.8,
                "group_c_avg_score": 0.6,
                "avg_score_gap": 0.2,
            },
        )

    def test_build_wrong_answer_rows_adds_reason_from_rank_and_top_prediction(self):
        details = [
            {
                "name": "B2-적색우회전녹색직진교차충돌",
                "expected": "차3-1",
                "top1": "차1-1",
                "rank": 999,
                "score": 0.42,
            },
            {
                "name": "C9-고속도로감속차로진입추돌",
                "expected": "차43-4",
                "top1": "차43-1",
                "rank": 3,
                "score": 0.71,
            },
        ]

        rows = build_wrong_answer_rows(details)

        self.assertEqual(rows[0]["cause"], "Top5 밖 오답")
        self.assertEqual(rows[1]["cause"], "유사 유형과 혼동")


if __name__ == "__main__":
    unittest.main()
