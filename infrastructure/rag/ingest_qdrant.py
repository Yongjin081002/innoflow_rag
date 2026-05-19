"""
과실비율 기준 chunks → Qdrant 저장
- 임베딩: embed_text (사고 상황 핵심만, 수정요소 제외)
- payload: id, content, embed_text, base_fault, modifiers, category
- 모델: boost_best_vlm_v4 (sentence-transformers)
"""
import os

from core import config

BASE_DIR = config.BASE_DIR
COLLECTION_NAME = config.QDRANT_COLLECTION
QDRANT_PATH = config.QDRANT_PATH


def get_model():
    from sentence_transformers import SentenceTransformer
    model_path = config.MODEL_PATH
    print(f"모델 로딩: {model_path}")
    model = SentenceTransformer(model_path)
    model.max_seq_length = 384
    return model


def insert_all():
    from infrastructure.rag.chunk_parser import parse_all_chunks
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct

    parsed = parse_all_chunks()
    rules = [p for p in parsed if p["is_rule"]]
    print(f"전체 {len(parsed)}건 중 규칙 {len(rules)}건 insert 대상")

    # 모델 로딩 & 임베딩
    model = get_model()
    embed_texts = [r["embed_text"] for r in rules]
    print(f"임베딩 중 ({len(embed_texts)}건)...")
    embeddings = model.encode(embed_texts, show_progress_bar=True, batch_size=32)
    dim = embeddings.shape[1]
    print(f"임베딩 완료 (dim={dim})")

    # Qdrant 클라이언트
    client = QdrantClient(path=QDRANT_PATH)

    # 기존 컬렉션 삭제
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in collections:
        client.delete_collection(COLLECTION_NAME)
        print(f"기존 '{COLLECTION_NAME}' 컬렉션 삭제")

    # 새 컬렉션 생성
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    print(f"'{COLLECTION_NAME}' 컬렉션 생성 (dim={dim})")

    # 포인트 생성 & 업서트
    points = []
    for i, (rule, vec) in enumerate(zip(rules, embeddings)):
        payload = {
            "id": rule["id"],
            "content": rule["content"],
            "embed_text": rule["embed_text"],
            "base_fault": rule["base_fault"],
            "modifiers": rule["modifiers"],
            "category": rule["category"],
        }
        points.append(PointStruct(
            id=i,
            vector=vec.tolist(),
            payload=payload,
        ))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"{len(points)}건 insert 완료")

    # 샘플 확인
    print(f"\n{'='*60}")
    print("샘플 3건 확인")
    print(f"{'='*60}")
    results = client.scroll(collection_name=COLLECTION_NAME, limit=3, with_vectors=False)
    for pt in results[0]:
        p = pt.payload
        print(f"\n  ID: {p['id']}")
        print(f"  Category: {p['category']}")
        print(f"  Embed text: {p['embed_text']}")
        print(f"  Base fault: {p['base_fault']}")
        print(f"  Modifiers: {len(p['modifiers'])}개")
        for m in p["modifiers"][:3]:
            print(f"    {m}")
        if len(p["modifiers"]) > 3:
            print(f"    ... (+{len(p['modifiers'])-3}개)")

    client.close()
    return len(points)


if __name__ == "__main__":
    insert_all()
