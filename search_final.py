"""
과실비율 기준 검색 (Qdrant + boost_best_vlm)
- 반환 형식: 공통 스키마 (type, id, content, base_fault, modifiers, category, source, score, metadata)
"""
import json
import os
import sys

import config

COLLECTION_NAME = config.QDRANT_COLLECTION


class FaultRuleSearcher:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        from qdrant_client import QdrantClient

        print(f"모델 로딩: {os.path.basename(config.MODEL_PATH)}")
        self.model = SentenceTransformer(config.MODEL_PATH)
        self.model.max_seq_length = 384

        if config.QDRANT_HOST:
            self.client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        else:
            self.client = QdrantClient(path=config.QDRANT_PATH)

    def search(self, query, top_k=5):
        """
        검색 결과를 공통 스키마로 반환

        Returns:
            list[dict]: 각 항목은 아래 구조
            {
                "type": "fault_rule",
                "id": "차1-1",
                "content": "원본 청크 텍스트",
                "base_fault": {"A": 0, "B": 100},
                "modifiers": [...],
                "category": "자동차",
                "source": "과실비율 인정기준",
                "score": 0.85,
                "metadata": {}
            }
        """
        query_vec = self.model.encode([query])[0].tolist()

        results = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=top_k,
            with_payload=True,
        )

        output = []
        for hit in results:
            p = hit.payload
            output.append({
                "type": "fault_rule",
                "id": p["id"],
                "content": p["content"],
                "base_fault": p["base_fault"],
                "modifiers": p["modifiers"],
                "category": p["category"],
                "source": "과실비율 인정기준",
                "score": round(hit.score, 4),
                "metadata": {
                    "embed_text": p["embed_text"],
                },
            })

        return output

    def close(self):
        self.client.close()


def main():
    """테스트 실행"""
    searcher = FaultRuleSearcher()

    queries = [
        "신호위반 직진 충돌",
        "야간 교차로 추돌",
        "주차장 출차 충돌",
    ]

    for query in queries:
        print(f"\n{'='*80}")
        print(f"  Query: \"{query}\"")
        print(f"{'='*80}")

        results = searcher.search(query, top_k=3)

        for i, r in enumerate(results, 1):
            print(f"\n  [{i}] {r['id']} (score: {r['score']})")
            print(f"      Category: {r['category']}")
            print(f"      Embed: {r['metadata']['embed_text']}")
            print(f"      Base fault: {r['base_fault']}")
            print(f"      Modifiers: {len(r['modifiers'])}개")
            for m in r["modifiers"][:3]:
                print(f"        {m['target']} {m['condition']}: {m['value']}")
            if len(r["modifiers"]) > 3:
                print(f"        ... (+{len(r['modifiers'])-3}개)")

    searcher.close()


if __name__ == "__main__":
    main()
