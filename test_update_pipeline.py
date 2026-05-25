import unittest

from update_pipeline import chunk_pages


class ChunkPagesTest(unittest.TestCase):
    def test_splits_only_standalone_rule_ids(self):
        pages = [
            (
                10,
                "제목\n차1-1\n(A) 녹색 직진\n사고 상황\n⊙ 차1-1 설명 문장\n차1-2\n(A) 황색 직진"
            ),
            (
                11,
                "후속 설명\n거1-1\n(A) 자전거 직진\n349 501 차43-1 대표유형에 통합"
            ),
        ]

        chunks = chunk_pages(pages)

        self.assertEqual([chunk["id"] for chunk in chunks], ["차1-1", "차1-2", "거1-1"])
        self.assertIn("=== 10페이지 ===", chunks[0]["content"])
        self.assertIn("⊙ 차1-1 설명 문장", chunks[0]["content"])
        self.assertNotIn("349 501 차43-1", [chunk["id"] for chunk in chunks])


if __name__ == "__main__":
    unittest.main()
